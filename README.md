<div align=center>
  
# MultiDampGen: A Self-Supervised Latent Diffusion Framework for Multiscale Energy-Dissipating Microstructure Generation
  
</div> 

<!-- 逆向设计 -->
* ## 🧭 **_Overview of the workflow_**
<div align=center>
  <img width="1000" src="Figs/Abstract.png"/>   
  <img width="1000" src="Figs/Workflow.png"/>
   <div align=center><strong>Fig. 1. The workflow of MultiDampGen framework</strong></div>
</div><br>    

* ## ⚛️ **_Datasets & Pre-trained models_**
  The multiscale microstructure dataset encompasses a total of *__50,000 samples__*. The dataset utilized in this study, along with the pre-trained weights of MultiDampGen, can be accessed through the link provided below.      

[**🔗The damping microstructure dataset**](https://github.com/AshenOneme/MultiDampGen/releases/tag/Dataset)     
[**🔗The weights of the MultiDampGen**](https://github.com/AshenOneme/MultiDampGen/releases/tag/Weights)

<div align=center>
  
## 🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟          

</div> 

<!-- T2C -->
* ## 🧱 **_TXT2CAE_**          
  The TXT2CAE plugin has been developed based on the ABAQUS-Python API, enabling the generation of three-dimensional finite element models from arbitrary patterns, along with automated mesh generation. Testing has demonstrated successful operation on *__ABAQUS versions 2018 to 2020__*.
<div align=center>
  <img width="1000" src="Figs/TXT2CAE.png"/>
   <div align=center><strong>Fig. 2. The TXT2CAE GUI</strong></div>
</div><br>   

<!-- Dataset -->
* ## 🏗️ **_Dataset_**          
  A total of 50,000 sets of microstructural data were extracted, including yield strength, yield displacement, first stiffness, and second-order stiffness. The distribution relationships were subsequently plotted based on *__volume fraction__* and *__physical scale__*.
<div align=center>
  <img width="1000" src="Figs/Dataset.png"/>
   <div align=center><strong>Fig. 3. Distribution of mechanical properties</strong></div>
</div><br>  

<!-- Architecture of MultiDampGen -->
* ## 🏗️ **_Architecture of MultiDampGen_**          
  A total of 50,000 sets of microstructural data were extracted, including yield strength, yield displacement, first stiffness, and second-order stiffness. The distribution relationships were subsequently plotted based on *__volume fraction__* and *__physical scale__*.
<div align=center>
  <img width="600" src="Figs/TopoFormer.png"/>
   <div align=center><strong>Fig. 4. Architecture of TopoFormer</strong></div>
    <img width="600" src="Figs/RSV.png"/>
   <div align=center><strong>Fig. 5. Architecture of RSV</strong></div>
    <img width="600" src="Figs/LDPM.png"/>
   <div align=center><strong>Fig. 6. Architecture of LDPM</strong></div>
</div><br>  

<!-- Generation process -->
* ## 🌆 **_Generation process_**          
  The generation process of multiscale microstructures is illustrated in the figure, with the *__red line__* representing the specified mechanical performance demands. The scales of the microstructures are randomly determined, and the generation results at each timestep are evaluated through finite element analysis. It can be observed that the hysteretic performance, indicated by the *__blue line__*, progressively approaches the target demands.
<div align=center>
  <img width="1000" src="Figs/GenerationProcess.gif"/>
   <div align=center><strong>Fig. 7. The generation process</strong></div>
</div><br> 

<!-- Generation results -->
* ## 🚀 **_Generation results_**          
  Regardless of how extreme the specified mechanical properties or scales may be, it is possible to generate microstructures that meet the demands. Additionally, by employing a latent diffusion approach, the generation efficiency has been improved significantly, achieving a square factor increase compared to the Denoising Diffusion Probabilistic Model (DDPM).
<div align=center>
  <img width="1000" src="Figs/Results.png"/>
   <div align=center><strong>Fig. 8. The generation results</strong></div>
</div><br> 
