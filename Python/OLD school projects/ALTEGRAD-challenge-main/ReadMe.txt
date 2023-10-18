------------------ Code description ------------------
Our code contains a total of 6 files:
-/ paper_representations.py: produce the paper embeddings of the abstracts.
-/ author_representations.py: produce the author embeddings and write it into a csv file.
-/ graph_representations.py: produce the graph embeddings using the Node2Vec model and write it into a csv file.
-/ graph_features.py: compute some features related to the graph and write it into a csv file.
-/ model.py: it contains the definition of our model.
-/ train_model.py: it's the file that trains the model and write the final predictions to the test_predictions.csv.
-/ roberta.py: produce paper embeddings based on the roBERTa model. (not used for the best submission)

------------------ Input Files / Intermediary Files ------------------
All files needs to be putted in the folder data.
Intermediairy files are found in the folder data.

------------------ Excution Order ------------------
To use the code follow the same sequence of excution.
We present for each file .py the needed input files that needs to be present in 
the folder data and it's output file.

*) Author embeddings: 
1) Run paper_representations.py: 
	input: abstract.txt 
	output: preproced_abstract.txt, paper_embeddings.txt
2) Run author_representations.py: 
	input: paper_embeddings.txt, author_paper.txt
	output: author_embedding.csv

*) Graph embeddings:
1) Run graph_representations.py:
	input: collaboration_network.edgelist
	output: graph_embedding.csv

*) Graph features:
1) Run graph_features.py:
	input: collaboration_network.edgelist
	output: graph_features.csv

*) Train model and predict:
1) Run train_model.py:
	input: graph_features.csv, graph_embedding.csv, author_embedding.csv, train.csv, test.csv
	output: test_predictions.csv

------------------ Notice ------------------
While using spider to run the code we couldn't use the code and had to run the file model.py then train_model.py.
However while using Visual Studio Code and the terminal it worked normally.