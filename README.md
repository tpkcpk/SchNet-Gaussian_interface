Usage: SchNet–Gaussian 16 Interface
To run calculations using the SchNet Neural Network Potential (NNP) through Gaussian 16, follow the steps below:

1. Prepare the Trained SchNet Model
Place your trained SchNet NNP model file (.pt or best_model) inside your working directory.

2. Set Up the Gaussian Input File (.gjf / .com)
In your Gaussian input file, specify the #external keyword pointing to the launcher script. Note that opt=nomicro is required and must not be omitted when running geometry optimizations:
#external='bash gausch.sh' opt=nomicro

Note: You may add other standard Gaussian keywords (e.g., freq, nosymm, or charge/multiplicity) depending on your calculation needs.

3. Configure Environment in gausch.sh
Edit gausch.sh to ensure it correctly sources the Python environment containing the required dependencies (PyTorch, ASE, SchNetPack, etc.):
source ~/script/python_env.sh

4. Adjust Preferences in gausch.py
Open gausch.py and modify the pref parameters (such as model paths, CPU thread limits, or unit conversions) according to your specific setup.

5. Run the Gaussian Calculation
Execute Gaussian 16 as usual:
g16 input.gjf output.log
