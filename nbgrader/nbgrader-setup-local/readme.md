# **Environment setup for nbgrader**

### This guide is crucial to get the basic nbgrader up and running, ***locally*** in your system. Essentially, this setup involves creating a new environment to avoid conflicting dependencies as well as to standardize the environments that collaborating instructors use in creating instructor-versions of Training notebooks.

---

#### **Current nbgrader in use**
- Version   : 0.6.1
- Build     : py37hc8dfbb8_0
- Channels  : conda-forge, defaults 

## **Methods to setup environment :-**

### **A. Create environment with packages automatically installed using .yml file**
1. In your terminal, navigate to the location of .yml file
2. Input the following :-

    `conda env create -f (filename).yml`

3. Verify the new environment is installed correctly with :-

    `conda env list`

4. The name of the environment should appear in the list.

### **B. Install using command line**
1. In your terminal, create a new environment 'nbgPy37' :-

    `conda create -n nbgPy37`

2. Install the necessary package with :-

    `conda install -c conda-forge nbgrader=0.6.1=py37hc8dfbb8_0`

## **How to verify if nbgrader is properly installed?**
1. Create a test course: _courseA_ with the following syntax :-

    `nbgrader quickstart courseA`

2. In the course folder, start jupyter notebook.

    `cd courseA`

    `jupyter notebook`

3. Click on this link, or copy and paste the link into your address bar. :-

    http://localhost:8888/notebooks/source/ps1/problem1.ipynb

4. If you see Kernel Error in your toolbar, you will need to input the following into a new instance of your terminal. :-

    `python -m ipykernel install --user`

5. Click on **'Validate'**. A success message box should appear.


