Integrated analysis platform of tissue histology and spatial transcriptomics at the **cell level**
========================================================

**Thor** enables cell-level spatial transcriptomics analysis by integrating histology and transcriptomic information through an anti-shrinking Markov diffusion method. 

Key features:

- Jointly leverages histological and transcriptomic information for accurate inference
- Achieves single-cell resolution without requiring additional single-cell reference data

.. image:: _static/workflow_part1.png
  :width: 100%
  :alt: workflow


Thor modules
========================================================

.. image:: _static/Thor_advanced_analysis_illustration.png
  :width: 100%
  :alt: function


What's new
========================================================
- 🚀 *2025-09-03*: Updated preprocessing scripts for Visium HD data and added a new tutorial in the `visiumhd/ <./visiumhd>`_ directory
- 🧬 Added full support for **Visium HD** data
- 🔗 Integrated **COMMOT** for cell–cell communication analysis

Installation
========================================================
Thor is written in Python (3.9+) and tested on macOS and Linux. We recommend installing it inside a virtual environment.

Step 1. Create a virtual environment (recommended)
--------------------------------------------------------

Using `conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: bash

   conda create -n Thor python=3.9
   conda activate Thor

Step 2. Clone the repository
--------------------------------------------------------

.. code-block:: bash

   git clone https://github.com/GuangyuWangLab2021/Thor.git
   cd Thor

Step 3. Install Thor
--------------------------------------------------------

- Base installation:

  .. code-block:: bash

     pip install .

- With optional visualization and advanced analysis:

  .. code-block:: bash

     pip install ".[vis,analysis]"

Cell–cell communication (COMMOT)
--------------------------------------------------------

Currently, Thor includes **COMMOT**, a state-of-the-art tool for spatial communication analysis.
This module is under active development, and we welcome feedback and contributions.

Requirements:

- R (tested with version 4.2.2)
- Python dependencies:

  .. code-block:: bash

     pip install --no-deps commot@git+https://github.com/biopzhang/COMMOT.git
     pip install karateclub@git+https://github.com/benedekrozemberczki/karateclub.git POT libpysal rpy2==3.5.11 anndata2ri==1.2

Usage
========================================================

- Visit the `Thor website <https://wanglab.tech/thor>`__ for API documentation and tutorials
- Launch the **Mjolnir web app** for interactive visualization of WSIs and Thor outputs: `https://wanglab.tech/mjolnir_launch/ <https://wanglab.tech/mjolnir_launch/>`__
- To reproduce the results in our paper, see the tutorials and parameters in the `parameters <./parameters/>`_ directory

Frequently Asked Questions
========================================================

*(Coming soon – please open an issue if you have questions you’d like us to cover!)*

Support
========================================================

Please report bugs, request features, or share feedback on the `GitHub Issues page <https://github.com/GuangyuWangLab2021/Thor/issues>`__.
