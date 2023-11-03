# EMProject

## Project
This is a repository of the Empirical Methods course project regarding analysis of Python code.  

## Data
We have collected more than a hundred reporitories containing mostly Python code. The links to the repositories may be found [here](MetricsCalculation/Repos/). A more detailed description of the process of data collection is described in the [Overleaf report](https://www.overleaf.com/project/65006cc84a750f7e0aa6a12c).  

[Repositories Collection](RepositoriesCollection/)  

[Metrics Collection](MetricsCollection/)  

[Metrics Values](data/)  

[Result of Analysis](MainTaskResultsAnalysis/)

## Analysis

We calculated all the above-described metrics for the repositories. For the SourceMeter, for each metric and repository, we opened the corresponding .csv file (either <repository\_name>-Class.csv or <repository\_name>-Method.csv) and took the average value for this metric. For Radon metrics, we did the same, but instead of .csv files we opened results.jsonl files corresponding to each repository. .csv tables with the metric values and the descriptive analysis of the class metrics can be found in our repository.  

The descriptive metrics such as count, mean, standard deviation, minimum, maximum, quantiles were collected and aggregated in the specific csv files corresponeding to methods and classes.  

A more detailed description of the process of data analysis is described in the [Overleaf report](https://www.overleaf.com/project/65006cc84a750f7e0aa6a12c). 

## Main task
The main task was identifying the minimal subset of metrics that explain the structure of the software repositories. For this, we tried two optimization strategies and two fitness functions:
* Particle Swarm Optimization (PSO) with Sammon error as a fitness function;
* Genetic Algorithms (GA) with Sammon error as a fitness function;
* PSO with Kruskal’s stress as a fitness function;
* GA with Kruskal’s stress as a fitness function.

A more detailed description of each stratagies and fitness functions are described in the [Overleaf report](https://www.overleaf.com/project/65006cc84a750f7e0aa6a12c).

## Team
* **Ninel Yunusova** - Data Collection, Modelling, Documents, Statistical Analysis
* **Georgy Andryushchenko** - Data Collection, Modelling, Documents, Statistical Analysis
* **Dinislam Gabitov** - Data Collection, Modelling, Statistical Analysis
* **Andrey Palaev** - Modelling, Documents
