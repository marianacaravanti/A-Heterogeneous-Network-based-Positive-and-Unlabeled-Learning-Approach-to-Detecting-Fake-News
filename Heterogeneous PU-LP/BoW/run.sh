#sudo apt install default-jre 
#sudo apt install openjdk-8-jre-headless
#pip3 install numpy scipy sklearn pandas networkx gensim nltk
#$1 - number of processes to run vector space models and matrices of PU-LP
#$2 - number of processes to run PU-LP ($1 x $2)
#Example: ./run.sh 2 4: Two processes will run vector space models and matrices of PU-LP. Eight processes will run PU-LP.
python3 create_scripts.py $1 $2
chmod -R 777 scripts
nohup ./scripts/fila_bow/fila_bow.sh &
