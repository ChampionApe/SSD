# Overview of models/classes

**US_EulerModel:**
    * ```US_EulerModel_c```: Defines base functions used in the Euler model for the US (no unemployment, only one pension structure parameter, no hand-to-mouth).
    * ```US_EulerModel_policy```: Defines classes used in identification of policy functions + LOG methods.
    * ```US_EulerModel_main```: Defines ```Model``` class that establishes solutions over time horizons, solves economic equilibria/steady state problems, and calibration methods.
    * Notebooks:
        * ```USEuler_Log.ipynb```: Calibrates relevant US model with log-preferences. The labor elasticity parameter is used to target the choice of $\kappa$
        * ```USEuler_Policy.ipynb```: Tests different methods for identification of policy functions - showcases syntax for changing initial guesses etc..
        * ```USEuler.ipynb```: Calibrate relevant US model model on grid of $\rho$.
        
**US:** *Note:* Not as developed as the ```US_EulerModel``` classes.
    * ```US_c```: Base functions used in US model with hand-to-mouth consumers (not unemployed).
    * ```US_policy```: Defines classes used in identification of policy functions.
    * ```US_main```: Defines ```Model``` class.
    * Notebooks:
        * ```US_PEE```: Calibrates PEE model and looks at exogenous changes in system characteristics.
        * ```US_Policy```: Tests different methods used in identification of policy functions.
        * ```US``` Calibrates ESC model on grids of $\rho$. *Note: Had issue with numerical stability here - migrated to EulerModel instead.*
        
**Older:** Models with unemployment and HtM consumers as per the Argentina implementation from working paper.  
    * ```logModel```: Log model PEE.
    * ```logModelESC```: Log model ESC.
    * ```CRRA```: CRRA model implementation (both PEE and ESC included here).
    
    