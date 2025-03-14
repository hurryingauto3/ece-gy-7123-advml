Assuming you’re in the root folder of your project, here are the commands you can run:
1.	Navigate to the Dockerfile location:

```
cd src/navsim
```

2.	Build the Docker image (tag it as “navsim”):

```
docker build -t navsim .
```

3.	Create a persistent data folder on your host (if not already created):

```
mkdir -p /path/to/navsim_data
```
Replace /path/to/navsim_data with your desired absolute path.

4.	Run the Docker container, mounting your persistent data folder:

```
docker run -it --rm -v /path/to/navsim_data:/navsim_workspace/dataset navsim
```


These commands will build the image from your Dockerfile and run it interactively. The -v option mounts your local data folder into the container so that once the data is downloaded, it will persist between runs.