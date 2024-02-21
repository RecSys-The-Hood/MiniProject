import numpy as np
import numpy.ma as ma

x = np.array([5,4,2,np.nan])
y = np.array([0,3,3,2])
weights = [1,4,7,8]

x_masked = ma.masked_invalid(x)
y_masked = ma.masked_invalid(y)

# Compute weighted means
weighted_mean_x = ma.average(x_masked, weights=weights)
weighted_mean_y = ma.average(y_masked, weights=weights)

weighted_covariance = ma.sum(weights * (x_masked - weighted_mean_x) * (y_masked - weighted_mean_y))
weighted_std_x = ma.sqrt(ma.sum(weights * (x_masked - weighted_mean_x)**2))
weighted_std_y = ma.sqrt(ma.sum(weights * (y_masked - weighted_mean_y)**2))

weighted_corr = weighted_covariance / (weighted_std_x * weighted_std_y)


print(weighted_corr)

# Compute weighted covariance
# weighted_covariance = np.sum(self.weights * (x - weighted_mean_x) * (y - weighted_mean_y))

# # Compute weighted standard deviations
# weighted_std_x = np.sqrt(np.sum(self.weights * (x - weighted_mean_x)**2))
# weighted_std_y = np.sqrt(np.sum(self.weights * (y - weighted_mean_y)**2))

# # Compute weighted Pearson's correlation coefficient
# if weighted_std_x == 0 or weighted_std_y == 0:
#     return 0  # Handle division by zero
# else:
#     weighted_corr = weighted_covariance / (weighted_std_x * weighted_std_y)
#     return weighted_corr

