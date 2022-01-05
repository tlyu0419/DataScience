# Recommendation Engine

Recommendation Engines are the programs which basically compute the similarities between two entities and on that basis, they give us the targeted output. If we look at the root of any recommendation engine, they all are trying to find out the amount of similarity between two entities. Then, the computed similarities can be used to deduce various kinds of recommendations and relationships between them.

**Recommendation Engines are mostly based on the following techniques:**

1. Popularity Based Filtering.
2. Collaborative Filtering (User Based / Item Based).
3. Hybrid User-Item Based Collaborative Filtering.
4. Content Based Filtering.

## Popularity Based Filtering

The most basic form of a recommendation engine would be where the engine recommends the most popular items to all the users. That would be generalized as everyone would be getting similar recommendations as we didn’t personalize the recommendations. These kinds of recommendation engines are based on the **Popularity Based Filtering**. The use case for this model would be the ‘Top News’ Section for the day on a news website where the most popular new for everyone is same irrespective of the interests of every user because that makes a logical sense because News is a generalized thing and it has got nothing to do with user’s interests.



## Collaborative Filtering 

In collaborative filtering, two entities collaborate to deduce recommendations on the basis of certain similarities between them. These filtering techniques are broadly of two types:



### User Based Collaborative Filtering

In user based collaborative filtering, we find out the similarity score between the two users. On the basis of similarity score, we recommend the items bought/liked by one user to other user assuming that he might like these items on the basis of similarity. This will be more clear when we go ahead and implement this. Major online streaming service, **Netflix** have their recommendation engine based on user based collaborative filtering.



### Item Based Collaborative Filtering

In item based collaborative filtering, the similarity of an item is calculated with the existing item being consumed by the existing users. Then on the basis of amount of similarity, we can say that if user X likes item A and a new item P is most similar to item A then it highly makes sense for us to recommend item P to user X.



### Hybrid User-Item Based Collaborative Filtering

This technique is basically a proper mixture of both the above techniques wherein the recommendations are not solely based on either. E-commerce websites like **Amazon** employ this technique to recommend item(s) to their customer.



### Content Based Filtering

In this technique, the users are recommended the similar content which they have used/watched/liked the most before. For example, if a user has been mostly listening to songs of similar type (bit rate, bps, tunes etc.), he will be recommended the songs falling under the same category decided based on certain features. The best example of this category would be **Pandora Radio** which is a music streaming and automated music recommendation internet radio service.

- Ref
  - [Building A Movie Recommendation Engine Using Pandas | by Nishit Jain | Towards Data Science](https://towardsdatascience.com/building-a-movie-recommendation-engine-using-pandas-e0a105ed6762)
  - [Similarity Measures — Scoring Textual Articles | by Saif Ali Kheraj | Towards Data Science](https://towardsdatascience.com/similarity-measures-e3dbd4e58660)