from typing import Any, Dict
    
import collections
import pickle
import weakref
import functools

class BaseHook:
    def __init__(self):
        self.weak_dict = weakref.WeakValueDictionary()
        self.strong_dict = {}

    def get(self, key: str) -> Any:
        try:
            return self.weak_dict[key]
        except KeyError:
            return self.strong_dict[key]
        
    def set(self, key: str, value: Any, weak: bool = True) -> None:
        if weak:
            self.weak_dict[key] = value
        else:
            self.strong_dict[key] = value

    def close(self):
        pass

    def after_diffusion_step(self):
        pass


class HookDispatcher(BaseHook):
    def __init__(self):
        super().__init__()
        self.hook_classes = {}
        self.enabled_hooks = {}

    def register_hook_class(self, hook_class: BaseHook) -> None:
        self.hook_classes[hook_class.name] = hook_class
        return hook_class

    def register_from_file(self, hooks_file: str) -> None:
        with open(hooks_file, "r") as f:
            env = {}
            exec(f.read(), env)
            for klass in env.values():
                if isinstance(klass, type) and issubclass(klass, BaseHook):
                    self.register_hook_class(klass)

    def enable(self, hook_name: str) -> None:
        if hook_name not in self.hook_classes:
            raise ValueError(f"Hook {hook_name} not found. Available hooks: {self.hook_classes.keys()}")
        self.enabled_hooks[hook_name] = self.hook_classes[hook_name]()

    def get(self, key: str) -> Any:
        raise NotImplementedError("HookDispatcher does not support get")
    
    def set(self, key: str, value: Any, weak: bool = True) -> None:
        for hook in self.enabled_hooks.values():
            hook.set(key, value, weak=weak)

    def set_dict(self, d: Dict[str, Any], weak: bool = True) -> None:
        for key, value in d.items():
            self.set(key, value, weak=weak)

    def close(self):
        for hook in self.enabled_hooks.values():
            hook.close()

    def after_diffusion_step(self):
        for hook in self.enabled_hooks.values():
            hook.after_diffusion_step()


HOOKS_DISPATCHER = HookDispatcher()

@HOOKS_DISPATCHER.register_hook_class
class SaveEverythingHook(BaseHook):
    name = "save_everything"

    def __init__(self):
        BaseHook.__init__(self)

    def after_diffusion_step(self):
        pass

