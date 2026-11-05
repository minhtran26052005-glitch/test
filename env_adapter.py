import copy
import hashlib
import json


def schema(properties, required):
    return {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}

S = {'type': 'string', 'minLength': 1}
MOCK_SCHEMAS = {
    'smart_home.get_status': schema({}, []),
    'smart_home.climate_set': schema({'device_id': S, 'temperature': {'type': 'number', 'minimum': 16, 'maximum': 30}}, ['device_id', 'temperature']),
    'ride.book': schema({'pickup': S, 'destination': S}, ['pickup', 'destination']),
    'sys.ask_missing_info': schema({'question': S}, ['question']),
}

class MockEnvironment:
    def __init__(self, case):
        self.state = copy.deepcopy(case.get('environment', {'devices': [{'device_id': 'ac_living', 'room': 'phòng khách', 'type': 'air_conditioner', 'temperature': 28}], 'rides': []}))
        self.schemas = {name: MOCK_SCHEMAS[name] for name in case.get('available_tools', list(MOCK_SCHEMAS))}

    def execute(self, name, arguments):
        if name == 'smart_home.get_status':
            return copy.deepcopy(self.state['devices'])
        if name == 'smart_home.climate_set':
            device = next((d for d in self.state['devices'] if d['device_id'] == arguments['device_id']), None)
            if device is None:
                return {'error': 'unknown device'}
            device.update(temperature=arguments['temperature'], power='on')
            return {'success': True, 'device': copy.deepcopy(device)}
        if name == 'ride.book':
            self.state.setdefault('rides', []).append(copy.deepcopy(arguments))
            return {'success': True, 'ride': copy.deepcopy(arguments)}
        if name == 'sys.ask_missing_info':
            return {'question': arguments['question']}
        raise ValueError(f'Unknown mock tool: {name}')

    def get_state_hash(self):
        return hashlib.sha256(json.dumps(self.state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get_state(self):
        return copy.deepcopy(self.state)


def make_environment(mode, case):
    if mode == 'mock':
        return MockEnvironment(case)
    if mode == 'official':
        return OfficialEnvironment(case)
    raise ValueError(f'ENV_MODE không hợp lệ: {mode}')

class OfficialEnvironment:
    """One in-memory Zalo devkit session, with no access to its underlying state by the agent."""
    def __init__(self, case):
        import importlib.util
        import sys
        import os
        from pathlib import Path
        root = Path(os.environ['DATA_ROOT']).resolve()
        split = os.environ['DATA_SPLIT']
        if split not in ('train', 'public_test'):
            raise ValueError(f'DATA_SPLIT không hợp lệ: {split}')
        package = root / split / 'zaic_tools'
        if not (package / '__init__.py').is_file():
            raise RuntimeError(f'Không tìm thấy devkit tại {package}')
        # Namespaced imports prevent train/public_test packages from colliding.
        module_name = f'_zaic_{split}'
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, package / '__init__.py', submodule_search_locations=[str(package)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        module = sys.modules[module_name]
        self.session = module.ToolEnv(split, case['id'], root=root)
        self.specs = {name: spec for name, spec in self.session.tools.items() if name in self.session.allowed}
        self.schemas = {name: spec.get('parameters', {'type': 'object'}) for name, spec in self.specs.items()}

    def validate(self, name, arguments):
        return self.session.validate(name, arguments)

    def execute(self, name, arguments):
        return self.session.call(name, arguments)

    def get_state_hash(self):
        return self.session.get_state_hash()

    def get_tool_calls(self):
        return self.session.get_tool_calls()

    def get_state(self):
        return self.session.get_state()
