import os
import subprocess

from configs.config import get_config

def install():
    save_dir = get_config()['tool_dir']
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    hifiasm_dir_name = f'hifiasm-0.18.8'
    if os.path.isfile(os.path.join(save_dir, hifiasm_dir_name, 'hifiasm')):
        print(f'\nFound hifiasm! Skipping installation...\n')
    else:
        # Install hifiasm
        print(f'\nInstalling hifisam...')
        subprocess.run(f'git clone https://github.com/chhylp123/hifiasm.git --branch 0.18.8 --single-branch {hifiasm_dir_name}', shell=True, cwd=save_dir)
        hifiasm_dir = os.path.join(save_dir, hifiasm_dir_name)
        subprocess.run(f'make', shell=True, cwd=hifiasm_dir)
      

    pbsim_dir_name = f'pbsim3'
    if os.path.isfile(os.path.join(save_dir, pbsim_dir_name, 'src', 'pbsim')):
        print(f'\nFound pbsim! Skipping installation...\n')
    else:
        # Install PBSIM3
        print(f'\nInstalling PBSIM3...')
        subprocess.run(f'git clone https://github.com/yukiteruono/pbsim3.git {pbsim_dir_name}', shell=True, cwd=save_dir)
        pbsim_dir = os.path.join(save_dir, pbsim_dir_name)
        subprocess.run(f'./configure; make; make install', shell=True, cwd=pbsim_dir)
       


if __name__ == '__main__':
    install()
