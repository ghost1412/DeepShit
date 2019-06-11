<h1> Keras-GPU Docker </h1>

<h2> How to Install </h2>
<p1> Install docker image using the Dockerfile using the command 
  
    docker build -t kerasdocker -f Dockerfile .

</p1>

<p1> Install GPU driver and then install [Nvidia docker](https://github.com/NVIDIA/nvidia-docker) </p1>

<p1> Then start this docker with Nvidia docker using the command given, you can also mount a folder as given below

    nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /mnt:/root/sharedfolder kerasdocker bash
  
</p1>
  
