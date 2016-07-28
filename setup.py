from setuptools import setup

requirements = ['docutils>=0.3',
                'setuptools',
                'numpy',
				'matplotlib>=1.4.2',
				'pyrfr=>0.2.1',
                'ConfigSpace']

setup(
	name='fanova',
	packages=['fanova'],
	version='1.0',
	install_requires=requirements,
	author='',
	author_email='wunschc@informatik.uni-freiburg.de',
	description = "Functional ANOVA: an implementation of the ICML 2014 paper 'An Efficient Approach for Assessing Hyperparameter Importance' by Frank Hutter, Holger Hoos and Kevin Leyton-Brown.",
	license = "FANOVA is free for academic & non-commercial usage. Please contact Frank Hutter		(fh@informatik.uni-freiburg.de) to discuss obtaining a license for commercial purposes."
)
