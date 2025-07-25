# MultiChoice_VisualDecisionMaking
# My Master's Project
At the Cognitive Science Lab led by Prof. Reza Ebrahimpour, and under the co-supervision of Dr. Saman Moghimi, I collaborate with a multidisciplinary team of computer science and cognitive science students to optimize and extend a pre-existing spiking convolutional neural network (SCNN). Our goal is to enhance this biologically inspired model for simulating multi-class decision-making processes, specifically aiming to mimic how the human brain categorizes natural objects through the ventral visual pathway—a key route in visual recognition.
This research bridges computational neuroscience and human cognition by developing a model that aligns closely with behavioral data obtained from psychophysical experiments. These experiments involve human participants performing multi-class visual forced-choice tasks while their eye movements are tracked. These experiments were fully designed and conducted by our research team in the lab. By varying the visual noise in natural image stimuli, we study the effects of sensory uncertainty on decision-making behavior.
The SCNN is designed to account for three key behavioral signatures: the distribution of reaction times, categorization accuracy (as reflected in psychometric functions), and confidence estimations derived from task difficulty. By integrating these components, our model aims to provide a mechanistic explanation of how the human brain processes complex visual stimuli under uncertainty.

**👉 Start here:** [linear_filter_snn_model.ipynb](./linear_filter_snn_model.ipynb)


The main required package for running these codes is SpykeTorch:
# Clone the repository
git clone https://github.com/miladmozafari/SpykeTorch

Alternatively, one can just run the following command
pip install git+https://github.com/miladmozafari/SpykeTorch.git

IMPORTANT: Current version of SpykeTorch does not support negative synaptic weights.

# The hierarchical two-module neurocomputational model for simulating object recognition:
<img width="2044" height="1150" alt="proposed_model_fig" src="https://github.com/user-attachments/assets/d32ba45d-64a8-4384-8dd6-642a35fc7d65" />
Link of some useful papers:
https://www.sciencedirect.com/science/article/abs/pii/S0893608017302903
https://www.sciencedirect.com/science/article/abs/pii/S0893608024002429
https://www.frontiersin.org/articles/10.3389/fnins.2019.00625/full

# ETH80 animal objects:
<img width="1806" height="628" alt="eth_animal_sample" src="https://github.com/user-attachments/assets/a9f18206-89b3-49ed-a475-7ee833b4aa9c" />
dataset:
https://github.com/chenchkx/ETH-80.git

