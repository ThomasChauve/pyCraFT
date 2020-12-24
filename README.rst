pyCraFT is build to analysed output from CraFT. It is not meant to be use easily as there is some trick to load the data.

The vtk output from CraFT are "strange". They do not open properly in paraview. Therefore I using vtk_split (software from CraFT installation) to be able to open the file. vtk_split need to be install on your computer and the path to it need to be given in the open function of runcraft.

May be this issue can be fix (see Herve ?)
