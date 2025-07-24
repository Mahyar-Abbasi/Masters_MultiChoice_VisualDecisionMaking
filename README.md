# MultiChoice_VisualDecisionMaking

At the Cognitive Science Lab led by Prof. Reza Ebrahimpour (my M.S. co-supervisor), I collaborate with a multidisciplinary team of computer science and cognitive science students to optimize and extend a pre-existing spiking convolutional neural network (SCNN). Our goal is to enhance this biologically inspired model for simulating multi-class decision-making processes, specifically aiming to mimic how the human brain categorizes natural objects through the ventral visual pathway—a key route in visual recognition.
This research bridges computational neuroscience and human cognition by developing a model that aligns closely with behavioral data obtained from psychophysical experiments. These experiments involve human participants performing multi-class visual forced-choice tasks while their eye movements are tracked. These experiments were fully designed and conducted by our research team in the lab. By varying the visual noise in natural image stimuli, we study the effects of sensory uncertainty on decision-making behavior.
The SCNN is designed to account for three key behavioral signatures: the distribution of reaction times, categorization accuracy (as reflected in psychometric functions), and confidence estimations derived from task difficulty. By integrating these components, our model aims to provide a mechanistic explanation of how the human brain processes complex visual stimuli under uncertainty.
For a quick overview of the model please check the local_linear_filter_snn_model.ipynb notebook.

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
