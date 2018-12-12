from setuptools import setup

setup(name='lazydiff',
      version='0.1',
      description='Simple automatic differentiation',
      url='https://github.com/CS207-Project-Group-7/cs207-FinalProject',
      author='Matteo Zhang, Joseph Davison, Raymond Lin, Zheng Yang',
      author_email='mzhangyb@gmail.com, josephddavison@gmail.com, rlin@college.harvard.edu, zhengyang@g.harvard.edu',
      license='MIT',
      packages=['lazydiff'],
      install_requires=['numpy'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-cov', 'scikit-learn'],)
