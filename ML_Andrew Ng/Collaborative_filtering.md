# Collaborative Filtering

Literally running linear regression for multiple users or object? Such that we can get parameters $\Theta$ for each of the linear regression model.

Definitions:
![CF varianble definitions](\images\CF.jfif)

We can get $\theta$ from known features $x$, or we can get the feature values from learning known $\theta$ of the users. Thus we get to simultaneously compute both.

![CF objective](images\CF_sim.jfif)

We use gradient descent to minimize the collaborative function.
The summations are litterally calculating for each of the $i,j$ pairs. In a movie example, it is equivalent to each user movie pair such that the user has rated the movie. We don't need to include $\theta_0,x_0$ like other algorithms since we are learning features so an extra feature with all 1s is not useful.

![CF objective](images\CF_vec.jfif)

vectorise as above.

---
## Extras

### Mean Noramalisation:

carry out mean normalisation for each ratings. This can deal with cases where a user has not rated any movies. This is beacuse, we have to add back the mean in prediction. Thus we are literally giving the average rating for a user without rating.  