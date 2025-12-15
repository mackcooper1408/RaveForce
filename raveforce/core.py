from urllib.request import urlopen

import datetime
import random
import struct
import sys

import gymnasium as gym
from gymnasium import spaces
import librosa
import numpy as np
import torch
from wasmer import (
    engine,
    Store,
    Module,
    Instance,
    Memory,
    ImportObject,
    Function,
    FunctionType,
    Type,
)
from wasmer_compiler_cranelift import Compiler
from audiobox_aesthetics.infer import initialize_predictor


def make(
    code="", target=[], total_step=16, step_len=0.125, criteria="raw", action_space=[]
):
    return Env(
        code=code,
        target=target,
        total_step=total_step,
        step_len=step_len,
        criteria=criteria,
        action_space=action_space,
    )


# end


def now():
    return datetime.datetime.now()


class Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        code="",
        target=[],
        total_step=16,
        step_len=0.125,
        criteria="raw",
        action_space=[],
    ):
        super().__init__()
        self.elapse_block = 0
        self.code = code
        self.target = np.array(target) if len(target) > 0 else np.array([0])
        self.total_step = total_step
        self.step_len = step_len
        self.criteria = criteria
        self.loaded = False
        self.step_count = 0

        # Initialize aesthetic predictor if needed
        self.predictor = None
        if self.criteria == "maa":
            self.predictor = initialize_predictor(ckpt=None)

        # some calculation
        self.para_num = code.count("{}")

        # Define observation space - continuous values for audio samples
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.target),), dtype=np.float32
        )

        # Define action space based on the provided action_space configuration
        if len(action_space) > 0:
            # Build Box action space from the action_space configuration
            low = []
            high = []
            for action in action_space:
                if action[0] == "lin" or action[0] == "exp":
                    low.append(action[1])
                    high.append(action[2])
                elif action[0] == "choose":
                    # For discrete choices, use the min and max values
                    low.append(min(action[1]))
                    high.append(max(action[1]))
                elif action[0] == "rel":
                    # For relative actions, use a reasonable range
                    low.append(-10.0)
                    high.append(10.0)
                else:
                    low.append(0.0)
                    high.append(1.0)
            self.action_space = spaces.Box(
                low=np.array(low, dtype=np.float32),
                high=np.array(high, dtype=np.float32),
                dtype=np.float32,
            )
            self._custom_action_space = action_space
        else:
            # Default to a single continuous action
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self._custom_action_space = []

    def reset(self, seed=None, options=None):
        # Set the seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Reset step counter
        self.step_count = 0

        # use wasmer-python
        if self.loaded:
            self.instance.exports.reset()
        else:
            import_object = ImportObject()
            store = Store(engine.JIT(Compiler))
            import_object.register(
                "env", {"now": Function(store, now, FunctionType([], [Type.F64]))}
            )
            # self.module = Module(self.store, open('glicol_wasm.wasm', 'rb').read())
            # self.module = Module(self.store, urlopen('https://cdn.jsdelivr.net/gh/chaosprint/glicol/js/src/glicol_wasm.wasm', 'rb').read())
            binary = urlopen(
                "https://cdn.jsdelivr.net/gh/chaosprint/glicol@latest/js/src/glicol_wasm.wasm"
            ).read()
            module = Module(store, binary)
            self.instance = Instance(module, import_object)
            self.loaded = True

        # empty_observation = np.zeros(int(self.step_len * self.total_step))
        self.audio = [[], []]
        empty_observation = np.zeros(len(self.target), dtype=np.float32)
        # print(self.action_space.actions)
        # if plot:
        #     plt.plot(empty_observation)

        info = {}
        return empty_observation, info

    def step(self, action):
        # Convert action to proper format if needed
        action = np.array(action).flatten()

        # Apply custom action space transformations if defined
        if len(self._custom_action_space) > 0:
            processed_action = self._process_action(action)
        else:
            processed_action = action

        inPtr = self.instance.exports.alloc(128)
        resultPtr = self.instance.exports.alloc_uint8array(256)

        self.audioBufPtr = self.instance.exports.alloc_uint8array(256)

        # send the code to the wasm
        code = self.code.format(*processed_action)
        # print(self.criteria)
        code = bytes(code, "utf-8")
        codeLen = len(code)
        codePtr = self.instance.exports.alloc_uint8array(codeLen)
        self.memory = self.instance.exports.memory.uint8_view(codePtr)
        self.memory[0:codeLen] = code

        self.instance.exports.update(codePtr, codeLen)

        audioBufPtr = self.instance.exports.alloc(256)

        # start the engine
        # self.instance.exports.run_without_samples(codePtr, codeLen)
        self.num_block = int(self.step_len * 44100 / 128)
        # self.elapse_block += self.num_block

        # self.audio = []

        for _ in range(self.num_block):

            self.instance.exports.process(inPtr, audioBufPtr, 256, resultPtr)
            bufuint8 = self.instance.exports.memory.uint8_view(offset=int(audioBufPtr))[
                :1024
            ]
            nth = 0
            buf = []
            while nth < 1024:
                byte_arr = bytearray(bufuint8[nth : nth + 4])
                num = struct.unpack("<f", byte_arr)
                buf.append(num[0])
                nth += 4
            result = self.instance.exports.memory.uint8_view(offset=resultPtr)
            result_str = "".join(map(chr, filter(lambda x: x != 0, result[:256])))
            # deprecated in 2022
            # self.instance.exports.process_u8(self.audioBufPtr)
            # self.memory = self.instance.exports.memory.uint8_view(self.audioBufPtr)
            # self.buf = [struct.unpack('<f', bytearray(
            #     self.memory[i: i+4]))[0] for i in range(0,256,4)]
            self.audio[0].extend(buf[:128])
            # self.audio[1].extend(buf[128:256])

        padded_observation = self.padding_to_total()
        reward = self.calc_reward(padded_observation)

        self.step_count += 1

        # Gymnasium requires separate terminated and truncated flags
        terminated = self.step_count >= self.total_step
        truncated = False  # Can be used for time limits or other truncation conditions

        # Note: result_str is for error/debug messages from WASM engine
        # Empty result_str is normal when there are no errors
        info = {
            "result": result_str if result_str != "" else "",
            "audio_samples": len(self.audio[0]),  # Debug: confirm audio is generated
        }

        return padded_observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def _process_action(self, action):
        """Process continuous action values based on custom action space configuration."""
        result = []
        for i, (act_val, act_config) in enumerate(
            zip(action, self._custom_action_space)
        ):
            if act_config[0] == "lin":
                # Linear scaling from normalized [low, high] to target range
                result.append(float(act_val))
            elif act_config[0] == "exp":
                # For exponential scaling, ensure value is in valid range
                result.append(float(act_val))
            elif act_config[0] == "choose":
                # For discrete choices, find nearest value
                choices = act_config[1]
                idx = int(np.clip(act_val, 0, len(choices) - 1))
                result.append(choices[idx])
            elif act_config[0] == "rel":
                # Relative action - would need context from previous actions
                result.append(float(act_val))
            else:
                result.append(float(act_val))
        return result

    def padding_to_total(self):
        # My implementation: pad or truncate to target length
        # pad_width = len(self.target) - len(self.audio[0])
        # padded = (
        #     np.pad(self.audio[0], (0, pad_width))
        #     if pad_width > 0
        #     else np.array(self.audio[0][: len(self.target)])
        # )

        # Original implementation: pad or truncate to target length
        padded = np.zeros(len(self.target))
        audio_len = min(len(self.audio[0]), len(self.target))
        padded[:audio_len] = self.audio[0][:audio_len]
        return padded

    def calc_reward(self, padded_observation):
        if self.criteria == "raw":
            # print(padded_observation)
            # print(len(padded_observation), len(self.target))
            # print((padded_observation - self.target))
            # mse = 0.1
            mse = (np.square(padded_observation - self.target)).mean(axis=0)
            return float(-mse)
        elif self.criteria == "mfcc":
            # Extract MFCC features from both audio signals
            sr = 44100  # Sample rate used in the audio engine

            # Extract MFCCs for observed and target audio
            obs_mfcc = librosa.feature.mfcc(y=padded_observation, sr=sr, n_mfcc=13)
            target_mfcc = librosa.feature.mfcc(y=self.target, sr=sr, n_mfcc=13)

            # Calculate L2 norm (Euclidean distance) between MFCCs
            l2_distance = np.linalg.norm(obs_mfcc - target_mfcc)
            return float(-l2_distance)
        elif self.criteria == "stft":
            # Short-Time Fourier Transform comparison
            sr = 44100  # Sample rate used in the audio engine

            # Compute STFT magnitude spectrograms
            obs_stft = np.abs(librosa.stft(padded_observation))
            target_stft = np.abs(librosa.stft(self.target))

            # Calculate L2 norm between spectrograms
            l2_distance = np.linalg.norm(obs_stft - target_stft)
            return float(-l2_distance)
        elif self.criteria == "cqt":
            # Constant-Q Transform comparison
            sr = 44100  # Sample rate used in the audio engine

            # Compute CQT
            obs_cqt = np.abs(librosa.cqt(padded_observation, sr=sr))
            target_cqt = np.abs(librosa.cqt(self.target, sr=sr))

            # Calculate L2 norm between CQT representations
            l2_distance = np.linalg.norm(obs_cqt - target_cqt)
            return float(-l2_distance)
        elif self.criteria == "maa":
            # Meta Audiobox Aesthetic quality assessment
            sr = 44100  # Sample rate used in the audio engine

            # Ensure audio is at least 1 second (16000 samples at 16kHz after resampling)
            if len(padded_observation) < 44100:
                padded_obs = np.pad(
                    padded_observation, (0, 44100 - len(padded_observation))
                )
            else:
                padded_obs = padded_observation

            # Convert to torch tensor with shape [1, num_samples]
            audio_tensor = torch.from_numpy(padded_obs.astype(np.float32)).unsqueeze(0)

            # Create batch in the format expected by audiobox_aesthetics
            batch = [{"path": audio_tensor, "sample_rate": sr}]

            # Get predictions for all 4 aesthetic axes: CE, CU, PC, PQ
            results = self.predictor.forward(batch)

            # Average all 4 aesthetic scores (higher is better)
            # CE: Complexity, CU: Clarity, PC: Pleasantness, PQ: Quality
            avg_score = sum(results[0].values()) / 4

            return float(avg_score)
        else:
            return 0.0
