---
title: "[Draft] Dynamic Time Warping Algorithm for trajectories similarity"
layout: post
date: '2020-08-12 16:00:00'
description: You’ll find this post in your `_posts` directory. Go ahead and edit it
  and re-build the site to see your changes.
img: trajectories.png
fig-caption: trajectories
tags:
- Distance
- Trajectories
- DTW
- Algorithms
- Mobility
---

The Dynamic Time Warping (DTW) algorithm is one of the most used algorithms to find similarities between two time series. Its goal is to find the optimal global alignment between two time series by exploiting temporal distortions between them. DTW algorithm has been first used to match signals in speech recognition. 
However, its scope of use is wider as it can be used in Time series of two-dimensional space to model trajectories. In the case of modeling and analyzing trajectories, the DTW algorithm stands out as a good way of computing similarity between trajectories.

> A **time series** is a serie of [data points](https://en.wikipedia.org/wiki/Data_point) indexed (or listed or graphed) in time order. Most commonly, a time series > is a [sequence](https://en.wikipedia.org/wiki/Sequence) taken at successive equally spaced points in time.

In this article, we implement the DTW algorithm for human mobility analysis to find similarities between trajectories. 

**Quick reminder**: A **spatial trajectory** is a sequence of points $ S = (…, sᵢ, …) $ where sᵢ is a tuple of longitude-latitude such as _sᵢ = (φⱼ, λᵢ)_. We denote the **trajectory length** by $n = <code>&#124;</code>S<code>&#124;</code>.$ We denote a **subtrajectory** of S as $S(i, ie) = S[i..ie]$, where $0 ≤ i < iₑ ≤ n - 1$. The identified pair of subtrajectories is called a **motif**.

We can extract trajectories from several data sources. Some of the data sources used for capturing human mobility traces are as follows.

- **Call Detail Records (CDRs)**. Mostly accessible by network operators, CDRs have the advantage of having a huge sample size due to the ubiquity of mobile phones. These datasets offer a tower level precision and for obvious privacy reasons, these datasets are not publicly available.
- **Location-Based Social Network (LBSN)**. With the increasing use of social networks, new sources of data have emerged. Social Network platforms like Facebook, Twitter or FourSquare provide geographic check-ins or geolocated publications. These data, mostly freely available, can be used as a basis to model human mobility trajectories. The main drawback of LBSN&#39;s dataset is the sparsity of the location, indeed, entries are set only when check-ins are made.
- **GPS data**. Compared to the data sources previously cited, GPS data offers high spatial and temporal precision and offer a good socle to analyze full trajectories such as car movements, people movements,… The drawback of GPS data is that it requires additional preprocessing due to errors when the GPS signal is noisy.

To implement the DTW algorithm, we can either use an LBSN dataset or raw GPS data. In our case, we need accurate location data to draw trajectories, so we are going to use a GPS dataset. Many GPS datasets are available freely on the internet. Most of these datasets are designed from real-world data collected over a period of time. As example, we have the popular Nokia and Geolife datasets. 

## Definitions
A **time series** is a serie of [data points](https://en.wikipedia.org/wiki/Data_point) indexed (or listed or graphed) in time order. Most commonly, a time series is a [sequence](https://en.wikipedia.org/wiki/Sequence) taken at successive equally spaced points in time.

A **spatial trajectory** is a sequence of points $ S = (…, sᵢ, …) $ where sᵢ is a tuple of longitude-latitude such as _sᵢ = (φⱼ, λᵢ)_. In the GIS world, a trajectory is represented as a [LineString](https://en.wikipedia.org/wiki/Polygonal_chain) associated witht an attribute of time.

We denote the **trajectory length** by $n$ = <code>&#124;</code>S<code>&#124;</code>.

We denote a **subtrajectory** of S as $S(i, ie) = S[i..ie]$, where $0 ≤ i < iₑ ≤ n - 1$. The identified pair of subtrajectories is called a **motif**.

In the language of GIS, therefore, a trajectory is represented as a LINESTRING feature together with an attribute representing time.

To implement the DTW algorithm, we can either use an LBSN dataset or raw GPS data. In our case, we need accurate locations to draw trajectories, so we are going to use a GPS dataset. Many GPS datasets are available freely on the internet. Most of these datasets were built from real-life data collected over a period of time. As example, we have the popular Nokia and Geolife datasets. The drawback of these datasets is their sparsity and size. The goal of this blogpost been to implement the DTW on two sub-trajectories, discovering a motif is not a priority.

For the testing purposes, we can use a sample of the [Geolife dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52367) which contains trajectories of 182 individuals collected during 3 years by a research team of Microsoft Research Asia. It has a total of 17,621 trajectories of about 1.2 million kilometres.

To analyze this sample dataset, we can use the Pandas library on Python. To better understand how a trajectory similarity algorithm works, we will compute the distance manually using the DTW algorithm.

## Requirements:

- Python ≥3.6
- Pandas library
- Geolife sample dataset, available on this [link](https://github.com/scikit-mobility/tutorials/raw/master/AMLD%202020/data/geolife_sample.txt.gz).
- Some motivation and a big smile 😃

Let&#39;s set up the tools and explore our dataset:

```python
import pandas as pd

# p and q represent our raw trajectories
p = "../trajectories/20081020134500.plt"
q = "../trajectories/20081023055305.plt"

df_p = pd.read_csv(p, sep=',')
df_p.head()

```
<div align="center">
	<figure>
  <img src="/assets/img/df.png">
  <figcaption>figure 1. Result of  df_p.head()</figcaption>
</figure>
</div>


As you can see on figure 1, we have 4 main attributes in our dataset: lat (latitude), lng(longitude), datetime and uid(User ID). The coordinates are expressed in decimal degree using the **WGS84** datum.

To compute the DTW, we will extract sub-trajectories of 2 users, namely u₁ and u₂. The identified pair of sub-trajectories is called a motif. To find a motif with the closest distance between two sub-trajectories, a straightforward approach is to compute recursively the distance between the trajectories and to keep the trajectories that meet a particular threshold.

Depending on the complexity of the technique used and on the size of the dataset, this operation can quickly escalate into a lengthy process.

The main goal of this article is to guide through the process of finding the similarity between two trajectories and to find the warp path between two time series that is optimal.

In order to find the similarity between two trajectories, we need to compute a distance matrix _ **dG** _. It can be considered as a multidimensional array mapping every point of _ **P** _ with _ **Q** _ by their real distance. To find the distance between two geographic points, we can use the **Harvesine formula** illustrated in the equation below:


<div align="center">
	<figure>
  <img src="/assets/img/haversine.png" width="50%">
  <figcaption>Equation 1. Haversine Formula used to calculate the great-circle distance between two points 1 and 2</figcaption>
</figure>
</div>


where _ **λ₁** _,_ **ϕ₁** _ and _ **λ₂** _,_ **ϕ₂** _ are the geographical longitude and latitude in radians of the two points 1 and 2, _ **Δλ** _ , _ **Δϕ** _ be their absolute differences¹.

To compute the distance between u1 and u2 using DTW, we can define a function distance that computes the ground distance between two points. Then by using the principle of dynamic programming, we can go through the matrix recursively until we get the final score which will represent the DTW between our two trajectories.

The equation to compute the DTW P and Q (respectively u1 and u2) is the following:

<div align="center">
	<figure>
  <img src="/assets/img/dtw.png" width="50%">
  <figcaption></figcaption>
</figure>
</div>

Let's implement the algorithm in Python. Even if Python is not the best programming language when it comes to object oriented programming, we will structure our code as much as we can.

First, we have to create a class that we are going to use. The first class should define a point, represented by longitude and latitude.
```python
class Point:
	def __init__(self, latitude, longitude):
		self.latitude = latitude
		self.longitude = longitude

def __str__(self):
	return "Point("+self.latitude+", "+self.longitude+")"
```

Then we will create a function that takes a point as input and returns the ground distance between the initial point (defined by self) and the point added as parameter.

```python
def get_distance(self, point2:Point):
	delta_lambda = math.radians(point2.latitude - self.latitude)
	delta_phi = math.radians(point2.longitude - self.longitude)
	a = math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2) + math.cos(math.radians(self.latitude)) \
	* math.cos(math.radians(point2.latitude)) * math.sin(delta_phi / 2) * math.sin(delta_phi / 2)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	distance = cls.R * c
	return distance
```

Now, we have all the prerequisites to implement the code and find the distance between two trajectories. With our minimalist code, we can represent a trajectory as a list of _**Point[]**_. We can represent the ground distance between trajectories in a matrix.



## Time complexity

For two trajectories N and M, the time complexity of the DTW algorithm can be presented as *O(N M)*. Assuming that N\&gt;M, the time complexity is determined by the highest time spent to run the computation, so in this case, time complexity of the algorithm will be *O(N²)*.

DTW algorithm is known to have a quadratic time complexity that limits its use to only small time series data sets¹.

To optimize the computational time required by the DTW algorithm, some techniques have been developed such as **PruneDTW**, **SparseDTW**, **FastDTW** and the **MultiscaledDTW**. These techniques are not covered in this article. 

## Drawbacks of DTW

DTW performs well for finding similarity between two trajectories if they are similar in most parts, but the main drawback of this algorithm is that it gives non-meaningful results when it comes to comparing two trajectories containing significant dissimilar portions.

## Comparison with other similarities measures

By matching each point of a trajectory to another, DTW algorithm gives good results with uniformly sampled trajectories. Meanwhile, with non-uniformly sampled trajectories, DTW adds up all distances between matched pairs².

Algorithms available for finding similarities between trajectories can be sorted by applying a trade-off between efficiency and effectiveness. Among the most efficient method in terms of performance, the Euclidean Distance ranks amongst the best. In fact, Euclidean Distance between two time series is simply the sum of the squared distances from _n_th point to the other. For comparing trajectories, Euclidean distance shows a great performance in terms of computational time, but its
main disadvantage for time series data is that its results are very unintuitive.

<!-- Even though DTW gives a good balance between precision and computational time, so 

Figure #fig\_numb shows the results of DTW and DFD given 3 trajectories. S\_a, S\_b (uniformly sampled) and S

-->

## References

1. [https://en.wikipedia.org/wiki/Haversine\_formula](https://en.wikipedia.org/wiki/Haversine_formula)
2. Salvador, S., &amp; Chan, P. (2007). Toward accurate dynamic time warping in linear time and space. _Intelligent Data Analysis_, _11_(5), 561–580.
