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
from urllib.request import urlopen
import numpy as np
import sys
import datetime
import struct, random
import gymnasium as gym
from gymnasium import spaces


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
                "https://cdn.jsdelivr.net/gh/chaosprint/glicol@v0.11.9/js/src/glicol_wasm.wasm"
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
        print("step {} in total {}".format(self.step_count, self.total_step))

        # Gymnasium requires separate terminated and truncated flags
        terminated = self.step_count >= self.total_step
        truncated = False  # Can be used for time limits or other truncation conditions

        info = {"result": result_str if result_str != "" else ""}

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
        # pad_width = len(self.target) - len(self.audio[0])
        # print(len(self.target), len(self.audio))
        # padded = np.pad(self.audio, pad_width, "constant", constant_values=0)
        # print(padded)
        padded = np.zeros(len(self.target), dtype=np.float32)
        padded[: len(self.audio[0])] = self.audio[0]
        return padded

    def calc_reward(self, padded_observation):
        if self.criteria == "raw":
            # print(padded_observation)
            # print(len(padded_observation), len(self.target))
            # print((padded_observation - self.target))
            # mse = 0.1
            mse = (np.square(padded_observation - self.target)).mean(axis=0)
            return float(-mse)
        else:
            return 0.0
