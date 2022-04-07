from setuptools import setup, find_packages


setup(name='UCR',
      version='1.0.0',
      description='',
      author='Hao Chen',
      author_email='hao.chen@inria.fr',
      url='https://github.com/chenhao2345/UCR',
      install_requires=[
          'numpy', 'torch==1.4.0', 'torchvision==0.5.0',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss-gpu==1.6.3'],
      packages=find_packages(),
      keywords=[
          'Person Re-identification',
          'Contrastive Learning',
          'Lifelong Learning'
      ])

