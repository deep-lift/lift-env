# Elevator-RL
Elevator 환경 Scene 파일은 \Elevator-RL\Assets\Elevator\elevator_rl.unity 를 열면됨

용량문제로 github에 Elevator-RL\Assets\ML-Agents 에 tensorflowsharp  plungins가 업로드가 안됨
tensorflowsharp Plugins 는 https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Basic-Guide.md#setting-up-ml-agents-within-unity
에 링크가 있으니 다운받아서 설치해야만 한다.



설치방법

1. Clone repo
git clone https://github.com/deep-lift/lift-env.git

2. Create anaconda virtual env
conda create -n unity_env python=3.6

3. Install depedencies (파이선 3.7이상일 경우 0.8.1 호환이 안되서 다운로드 불가)
pip install mlagents==0.8.1
pip install mlagents_envs==0.8.1
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install gym
pip install tensorflow 
pip install tensorboardX


다음과 같은 오류가 떨어질경우
ModuleNotFoundError: No module named 'mlagents.envs'

madqn.py에 UnityEnvironment import 부분을 아래와 같이 수정
from mlagents.envs.environment import UnityEnvironment


C:\Users\SDS-1\anaconda3\envs\unity\python.exe D:/dev/workspace/lift-env/madqn.py
Traceback (most recent call last):
  File "D:/dev/workspace/lift-env/madqn.py", line 212, in <module>
    env = UnityEnvironment(file_name=env_name)
  File "C:\Users\SDS-1\anaconda3\envs\unity\lib\site-packages\mlagents\envs\environment.py", line 76, in __init__
    aca_params = self.send_academy_parameters(rl_init_parameters_in)
  File "C:\Users\SDS-1\anaconda3\envs\unity\lib\site-packages\mlagents\envs\environment.py", line 538, in send_academy_parameters
    return self.communicator.initialize(inputs).rl_initialization_output
  File "C:\Users\SDS-1\anaconda3\envs\unity\lib\site-packages\mlagents\envs\rpc_communicator.py", line 80, in initialize
    "The Unity environment took too long to respond. Make sure that :\n"
mlagents.envs.exception.UnityTimeOutException: The Unity environment took too long to respond. Make sure that :
	 The environment does not need user interaction to launch
	 The Academy's Broadcast Hub is configured correctly
	 The Agents are linked to the appropriate Brains
	 The environment and the Python interface have compatible versions.


