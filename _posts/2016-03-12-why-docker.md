---
layout: post
title: Why Docker?
---

Containers are in use for long time. They provide easy way to run applications in isolated light weight process. 



**Why Docker Instead of VMs?**
running 30 instances of applcation each on seperate VM involves
1. copying complete VM image
2. booting 
3. running full kernel
4. running actual application

container share all services from host OS
1. create container
2. run application

Saves a lot of processing power and time while initial spin up of application instance as well as throughout life of application.

http://bodenr.blogspot.com/2014/05/kvm-and-docker-lxc-benchmarking-with.html


easy to move from cloud to cloud


How it help speed up developement?


**Is it just another buzzword?**
Docker is going to stay. The industry is creating major investments in the technology all across the board with skyrocketing adoption in the works.