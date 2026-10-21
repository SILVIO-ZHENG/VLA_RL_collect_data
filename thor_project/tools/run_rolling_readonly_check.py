"""Bounded real-model/G2 observation test; never installs or executes hardware actions."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexus_deploy.deployment import Deployment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fps', type=int, choices=(15, 30), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remote-directory', default='/home/agi/.cache/nexus-rolling-benchmark-b6Qhp9')
    p.add_argument('--seconds', type=int, default=50)
    p.add_argument('--config-json', type=Path, help='Explicit saved config; does not require the web UI')
    p.add_argument('--inference-mode', choices=('sync', 'async', 'rtc'), default='async')
    p.add_argument('--effector-control-mode', choices=('device-position', 'software-trajectory'))
    a = p.parse_args()
    if not 30 <= a.seconds <= 180:
        p.error('seconds must be in [30, 180]')
    if not a.remote_directory.startswith('/home/agi/.cache/nexus-') or '..' in Path(a.remote_directory).parts:
        p.error('only isolated Nexus cache directories may be tested')
    a.output.mkdir(parents=True, exist_ok=False)
    d = Deployment(Path('deploy.toml'))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if a.config_json is not None:
        config = json.loads(a.config_json.read_text())
    else:
        with opener.open('http://192.168.110.140:8765/api/bootstrap', timeout=10) as response:
            config = json.load(response)['defaults']
    config.update(hardware_mode='read-only', model_id='pi05', server_mode='reuse',
                  execution_fps=a.fps, servo_fps=100, duration_s=a.seconds, inference_mode=a.inference_mode)
    if a.effector_control_mode is not None:
        config['effector_control_mode'] = a.effector_control_mode
    config['safety']['workspace_safety_enabled'] = True
    if not config['safety'].get('safety_profile_id'):
        raise ValueError('A calibrated safety_profile_id is required for read-only validation')
    config.pop('_platform_telemetry_token', None)
    config.pop('_platform_telemetry_endpoint', None)
    checks = d.require_robot_read_only_ready(d.robot)
    (a.output/'preflight.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    d.ensure_robot_client_idle(d.robot)
    robot = replace(d.robot, remote_directory=a.remote_directory)
    session = uuid.uuid4().hex
    d.deploy_generated_safety_profile(robot, config['safety']['safety_profile_id'], session)
    command, remote_logs = d.robot_session_command(robot, d.models['pi05'], session, config)
    if '--rolling-safety' not in command:
        command += ['--rolling-safety', '--rolling-geometry=legacy']
    assert '--read-only' in command and '--execute' not in command
    (a.output/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (a.output/'command.json').write_text(json.dumps(command, ensure_ascii=False, indent=2))
    # SIGTERM is handled by the owned client's main/finally; vendor services
    # and model server are neither signalled nor restarted.
    launch = 'mkdir -p '+shlex.quote(remote_logs)+'; exec '+shlex.join(
        ['timeout', '--foreground', '--signal=TERM', '--kill-after=10s', f'{a.seconds+20}s', *command])
    print('READ_ONLY_START', session, flush=True)
    result = subprocess.run([*d.ssh_base(robot), launch], capture_output=True, text=True, timeout=a.seconds+40)
    (a.output/'console.log').write_text(result.stdout+'\n'+result.stderr)
    subprocess.run(['rsync', '-a', '-e', d.rsync_shell(robot),
                    robot.ssh_target+':'+remote_logs+'/', str(a.output)+'/' ], check=True, timeout=40)
    summary = dict(session=session, remote_logs=remote_logs, ssh_returncode=result.returncode,
                   intentional_timeout=result.returncode == 124, hardware_mode='read-only')
    (a.output/'result.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if result.returncode not in (0,124):
        print(result.stderr[-5000:])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
