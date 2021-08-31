#! /usr/bin/python3

import argparse
from pathlib import Path
import subprocess
import yaml


def main():
    parser = argparse.ArgumentParser(description='Setup and connect to experimentation GCP instance')
    parser.add_argument('cmd', choices=['start', 'stop', 'delete'])
    parser.add_argument('project_folder', type=Path,
                        help='Project folder. Must be the name of a project sub folder under this repository')
    args = parser.parse_args()
    gcp_resource_tagname = args.project_folder.name.replace('_', '-')

    with (Path(args.project_folder) / ".gcp/config.yaml").open() as f:
        config = yaml.safe_load(f)

    def zone_project_args():
        v = []
        if 'zone' in config: v.append(f'--zone={config["zone"]}')
        if 'project' in config: v.append(f'--zone={config["project"]}')
        return v

    def start_instance():
        try:
            subprocess.check_output([
                'gcloud', 'compute', 'instances', 'create', f'nnexp-vm--{gcp_resource_tagname}',
                f'--machine-type={config["machine_type"]}',
                '--preemptible',
                '--network-interface=network-tier=PREMIUM,subnet=default',
                "--metadata=framework=PyTorch:1.9,google-logging-enable=0,google-monitoring-enable=0,install-nvidia-driver=True,shutdown-script=/opt/deeplearning/bin/shutdown_script.sh,status-config-url=https://runtimeconfig.googleapis.com/v1beta1/projects/sensor-experiments/configs/deeplearning-1-config,status-uptime-deadline=600,status-variable-path=status,title=PyTorch/CUDA11.0.GPU,version=75,startup-script=pip\ install\ pytorch_lightning,startup-script-url=gs://nnexp-bucket/shutdown_on_idle.sh",
                '--no-restart-on-failure',
                '--maintenance-policy=TERMINATE',
                '--service-account=508309808341-compute@developer.gserviceaccount.com',
                '--scopes=https://www.googleapis.com/auth/cloud.useraccounts.readonly,https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/cloudruntimeconfig,https://www.googleapis.com/auth/compute',
                f'--accelerator={config["accelerator"]}',
                '--tags=deeplearning-1-deployment,deeplearning-vm',
                '--image=pytorch-1-9-cpu-v20210714-debian-10',
                '--image-project=click-to-deploy-images',
                '--boot-disk-size=100GB',
                '--boot-disk-type=pd-standard',
                '--boot-disk-device-name=tensorflow-vm-tmpl-boot-disk',
                '--no-boot-disk-auto-delete',
                '--no-shielded-secure-boot',
                '--shielded-vtpm',
                '--shielded-integrity-monitoring',
                '--labels=goog-dm=deeplearning-1',
                '--reservation-affinity=any'] + zone_project_args())
        except subprocess.CalledProcessError as e:
            if b'already exists' in e.output:
                pass
            else: raise

        subprocess.check_output(['gcloud', 'compute', 'instances', 'start', f'nnexp-vm--{gcp_resource_tagname}'] + zone_project_args())
        subprocess.check_output(['gcloud', 'compute', 'config-ssh'])
        print('Machine info added to `~/.ssh/config`')

    def stop_instance():
        subprocess.check_output(['gcloud', 'compute', 'instances', 'stop', f'nnexp-vm--{gcp_resource_tagname}'] + zone_project_args())

    def delete_instance():
        subprocess.check_output(['gcloud', 'compute', 'instances', 'delete', f'nnexp-vm--{gcp_resource_tagname}'] + zone_project_args())

    dict(start=start_instance,
         stop=stop_instance,
         delete=delete_instance)[args.cmd]()

if __name__ == '__main__':
    main()

# Notes:
# pip install pytorch_lightning boto3
# sudo-apt get install aws-cli
# sudo apt-get install aws-cli