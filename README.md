# Wine-prediction
## AWS SPARK Cluster Creation:
 1. Click "Create new instance" after navigating to the EMR interface.
 2. Give your cluster a name.
 3. Switch from automatic to manual cluster termination.
 4. In the security configuration, provide the key_pair, which is a .pem file.
 5. Since we need to run on four EC2 instances, set one for core and four for tasks when creating the instance.
 6. Next, choose the IAM roles' default roles.
 
 7. To upload the Python and CSV files, create an S3 queue.
 s3://predofwine
 
 8. To connect to the cluster, open a terminal and enter the command below.
 Hadoop@ec2-52-201-250-228.compute-1.amazonaws.com ssh -i ~/Hw2BigData.pem

## Using Docker to execute
 1. To switch users, type "sudo su."
 2. Use "pip install numpy --user" to install numpy.
 3. Run "spark-submit s3://predofwine/training.py" after that. It generates an ML model 
 4. After running the file from the S3 bucket. "spark-submit s3://predofwine/prediction.py s3://predofwine/ValidationDataset.csv" should then be executed. It makes use of the developed model, verifies the information from the CSV file, and outputs the outcome.  

## Execution with Docker
 1. Start the docker in the EC2 by running the commands below.
 2. To start Docker, sudo systemctl
 3. System Ctl: sudo enable Docker

To obtain the image from the Docker repository, use the command below.

 1. Docker pull rohitha3/predofwine:train with sudo
 2. Extract rohitha3/predofwine:predict using sudo docker pull
 
 3. Use the following command to run the train tag picture in order to generate an ML model. run -v /home/ec2-user/:/job rohitha3/predofwine:train with sudo docker
 4. Use the predict tag on the image to determine its accuracy.
 <>  Run -v /home/ec2-user/:/job rohitha3/predofwine:predict ValidationDataset.csv using sudo docker.
