This application is run inside a Docker container that manages installs and requirements needed to run the code.  With Docker and docker-compose installed, the entire process can be run by calling:

docker-compose run dev python -c "import runctl; runctl.run()"

Some exploratory work / analysis for presentation was done in ipython notebooks.  To launch a jupyter notebook server on your localhost at port 8888 run (a link in the console will tell you where to navigate):

docker-compose up jupyter

