#ifndef ZIPF_H__
#define ZIPF_H__
#include <bits/stdc++.h>
using namespace std;
#define TRUE 1
#define FALSE 0

int zipf(double alpha, int n, double z) {
  static int first = TRUE;   // Static first time flag
  static double c = 0;       // Normalization constant
  static double *sum_probs;  // Pre-calculated sum of probabilities
  static double lst_alpha = -1e9;
//   double z;                  // Uniform random number (0 < z < 1)
  int zipf_value;            // Computed exponential value to be returned
  int i;                     // Loop counter
  int low, high, mid;        // Binary-search bounds

  // Compute normalization constant on first call only
  if (first == TRUE || lst_alpha != alpha) {
    lst_alpha = alpha;
    for (i = 1; i <= n; i++) c = c + (1.0 / pow((double)i, alpha));
    c = 1.0 / c;

    sum_probs = (double *)malloc((n + 1) * sizeof(*sum_probs));
    sum_probs[0] = 0;
    for (i = 1; i <= n; i++) {
      sum_probs[i] = sum_probs[i - 1] + c / pow((double)i, alpha);
    }
    first = FALSE;
  }

  // Pull a uniform random number (0 < z < 1)
//   do {
//     z = dis(genrd);
//   } while ((z == 0) || (z == 1));

  // Map z to the value
  low = 1, high = n;
  do {
    mid = floor((low + high) / 2);
    if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) {
      zipf_value = mid;
      break;
    } else if (sum_probs[mid] >= z) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  } while (low <= high);

  // Assert that zipf_value is between 1 and N
  assert((zipf_value >= 1) && (zipf_value <= n));

  return (zipf_value - 1);
}

#endif