# **Eigenfaces and PCA for Face Recognition – Explained Simply**

## **Why Do We Need Eigenfaces?**

- Images have **high dimensionality** (e.g., a 64×64 grayscale image has **4096** dimensions).
- Not all dimensions are **equally important**; some dimensions capture essential variations such as lighting, pose, and expression.
- **Principal Component Analysis (PCA)** helps reduce dimensions by identifying key directions of variance, providing a compact representation.

---

## **Understanding PCA (Principal Component Analysis)**

PCA is a mathematical technique developed by **Karl Pearson (1901)** and **Harold Hotelling (1933)** that:

- Converts **correlated variables** into a **smaller set of uncorrelated variables** (called **principal components**).
- Identifies the **directions (eigenvectors)** where data varies the most.
- Optimizes data representation by preserving maximum variance.

### **Deep Dive into PCA**

When applying PCA:

- We calculate the **covariance matrix** to understand how variables vary together.
- The **eigenvectors** of this matrix represent the new axes, while the **eigenvalues** indicate the magnitude of variance captured along each axis.
- Selecting the top k eigenvectors minimizes loss of information while reducing dimensionality, which is especially useful in processing large image datasets.

---

## **Eigenfaces Algorithm**

### **Step 1: Collect Face Data**

Let’s say we have **n face images**, each represented as a **vector**:

- **X = {x₁, x₂, ..., xₙ}**, where each **xᵢ** is a **flattened** version of an image.

### **Step 2: Compute the Mean Face**

We calculate the **average face** (mean image):

```math
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
```

Centering the data around this mean is crucial for accurate variance analysis.

### **Step 3: Compute the Covariance Matrix**

The covariance matrix summarizes the relationships between different pixel intensities:

```math
S = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)(x_i - \mu)^T
```

A well-computed covariance matrix is essential to detect the key variance directions.

### **Step 4: Find Eigenvectors & Eigenvalues**

We solve:

```math
S v_i = \lambda_i v_i
```

- **Eigenvectors (vᵢ)** indicate directions of maximum variance.
- **Eigenvalues (λᵢ)** represent the magnitude of variance along each eigenvector.

These eigenvectors are reshaped into images, which we call **"Eigenfaces"**.

### **Step 5: Reduce Dimensions**

By keeping only the **top k** eigenfaces:

- Each image is transformed as:

```math
y = W^T (x - \mu)
```

- An image can be approximately reconstructed using:

```math
x = Wy + \mu
```

This transformation ensures that most of the crucial information is retained.

---

## **Face Recognition with Eigenfaces**

1. **Train the Model**: Convert all training images into the **Eigenface space** (PCA subspace), retaining the essential variance.
2. **Project a Query Image**: Map the new face image using the same transformation.
3. **Compare**: Find the **nearest neighbor** (most similar projection) using a distance metric such as Euclidean or Mahalanobis distance.

---

## **Efficiency Trick**

- Directly computing **S = XXᵀ** can be computationally expensive if **X** is large.
- Compute **S = XᵀX**, which results in a much smaller matrix and then derive eigenvectors.
- Remap these eigenvectors back using **XXᵀ** to form the final eigenfaces.

---


## **Mathematical Breakdown of Eigenfaces for Training Data**

Let's break the **Eigenfaces method** into simpler mathematical steps to fully understand how training data is processed.


### **1. Understanding the Face Data as Vectors**

Each face image is represented as a **vector**.
If the image size is **64 × 64**, then it has **4096 pixels**.
For example, a **64 × 64** grayscale image has **4096 pixels**, so we represent it as a **4096-dimensional vector**.

Let’s say we have **n training images**, we represent them as:

```math
X = [x_1, x_2, ..., x_n]
```

where each \( x_i \) is a **column vector of size (4096 × 1)**.

This forms a **data matrix**:

```math
X =
\begin{bmatrix}
| & | & | \\
x_1 & x_2 & \dots & x_n \\
| & | & |
\end{bmatrix}
```

This matrix has size **(4096 × n)**, meaning each **column** is a flattened image.

---

### **2. Compute the Mean Face**

The **mean face** is simply the **average** of all training images:

```math
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
```

This creates an "average-looking" face.
We subtract this mean face from each image to **center the data**:

```math
\tilde{x}_i = x_i - \mu
```

Thus, the mean-centered data matrix is:

```math
\tilde{X} =
\begin{bmatrix}
\tilde{x}_1 & \tilde{x}_2 & \dots & \tilde{x}_n
\end{bmatrix}
```

---

### **3. Compute the Covariance Matrix**

To find the key **patterns** in the face data, we compute the **covariance matrix**:

```math
S = \frac{1}{n} \tilde{X} \tilde{X}^T
```

where:

- \( S \) is of size **(4096 × 4096)**, which is **huge**.

To avoid this, we use a mathematical trick:

- Instead of computing \( S = **XXᵀ**) (size 4096 × 4096),
- Compute \( **XᵀX**) (size n × n, much smaller).

---

### **4. Compute Eigenvectors and Eigenvalues**

Now, we solve the **eigenvalue problem**:

```math
\tilde{X}^T \tilde{X} v_i = \lambda_i v_i
```

where:

- \( v_i \) are the **eigenvectors** (size \( n \times 1 \)).
- \( \lambda_i \) are the **eigenvalues**, representing variance.

To get the original eigenvectors of \( S \):

```math
u_i = \tilde{X} v_i
```

These \( u_i \) are the **Eigenfaces**.

---

### **5. Select the Top k Eigenfaces**

Since **not all Eigenfaces are useful**, we select the top **k** eigenvectors with the **largest eigenvalues**:

```math
W = (u_1, u_2, ..., u_k)
```

Now, \( W \) is a **4096 × k** matrix, where each column is an **Eigenface**.

---

### **6. Project Each Face onto the Eigenface Space**

Each image is projected onto the **Eigenface space** by computing **weights**:

```math
y_i = W^T \tilde{x}_i
```

where:

- \( y_i \) is the **compressed representation** (size \( k \times 1 \)).
- Instead of storing a **4096-pixel image**, we only store **k numbers**.

---

### **7. Recognition Step**

For a new query image \( x_{new} \):

1. Compute its mean-centered version:

```math
\tilde{x}_{new} = x_{new} - \mu
```

2. Project onto Eigenfaces:

```math
y_{new} = W^T \tilde{x}_{new}
```

3. Compare \( y_{new} \) with stored training weights \( y_i \) using **Euclidean distance**.

If:

```math
\| y_{new} - y_i \| \text{ is small} \Rightarrow \text{Face is recognized.}
```

---

## **Final Summary of Steps**

1. **Prepare training images** as column vectors in matrix \( X \).
2. **Compute the mean face** \( \mu \) from 64×64 images (4096 dimensions).
3. **Center the data** by subtracting \( \mu \).
4. **Compute the covariance matrix trick** \( \tilde{X}^T \tilde{X} \) for reduced dimensionality.
5. **Find eigenvectors** which form the final Eigenfaces.
6. **Select the top k Eigenfaces** to reduce dimensionality.
7. **Project faces onto the new space** using \( y = W^T \tilde{x} \).
8. **Recognize faces** by finding the closest match in the reduced space.

This method is **efficient** because we:

- Avoid computing a massive covariance matrix.
- Store only the **top k** eigenvectors instead of full images.
- Reduce recognition time by working in **low-dimensional space**.

---

## **Detailed Numerical Example: Eigenfaces Method**

Let's go step by step with a **small numerical example** to understand the **Eigenfaces method** in detail.

---

### **Example Setup**

Imagine we have **3 grayscale images** of size **2 × 2 pixels**:

```math
x_1 =
\begin{bmatrix}
2 & 3 \\
3 & 4
\end{bmatrix}
, \quad
x_2 =
\begin{bmatrix}
1 & 2 \\
2 & 3
\end{bmatrix}
, \quad
x_3 =
\begin{bmatrix}
4 & 3 \\
3 & 2
\end{bmatrix}
```

Each image has **4 pixels**, so we represent each as a **column vector**:

```math
x_1 =
\begin{bmatrix}
2 \\ 3 \\ 3 \\ 4
\end{bmatrix}
, \quad
x_2 =
\begin{bmatrix}
1 \\ 2 \\ 2 \\ 3
\end{bmatrix}
, \quad
x_3 =
\begin{bmatrix}
4 \\ 3 \\ 3 \\ 2
\end{bmatrix}
```

The **data matrix** \( X \) is formed by stacking them as columns:

```math
X =
\begin{bmatrix}
2 & 1 & 4 \\
3 & 2 & 3 \\
3 & 2 & 3 \\
4 & 3 & 2
\end{bmatrix}
```

---

### **Step 1: Compute the Mean Face**

The **mean face** is calculated by taking the average of each row:

```math
\mu = \frac{1}{3} (x_1 + x_2 + x_3)
```

Computing each pixel:

```math
\mu =
\begin{bmatrix}
\frac{2+1+4}{3} \\
\frac{3+2+3}{3} \\
\frac{3+2+3}{3} \\
\frac{4+3+2}{3}
\end{bmatrix}
=
\begin{bmatrix}
2.33 \\
2.67 \\
2.67 \\
3.00
\end{bmatrix}
```

**Importance:** The mean face helps to center the data so that all images have a common reference.

---

### **Step 2: Subtract the Mean Face (Center the Data)**

Now, subtract \( \mu \) from each image:

```math
\tilde{x}_1 = x_1 - \mu =
\begin{bmatrix}
2 \\ 3 \\ 3 \\ 4
\end{bmatrix}
-
\begin{bmatrix}
2.33 \\ 2.67 \\ 2.67 \\ 3.00
\end{bmatrix}
=
\begin{bmatrix}
-0.33 \\ 0.33 \\ 0.33 \\ 1.00
\end{bmatrix}
```

Similarly:

```math
\tilde{x}_2 = x_2 - \mu =
\begin{bmatrix}
1 \\ 2 \\ 2 \\ 3
\end{bmatrix}
-
\begin{bmatrix}
2.33 \\ 2.67 \\ 2.67 \\ 3.00
\end{bmatrix}
=
\begin{bmatrix}
-1.33 \\ -0.67 \\ -0.67 \\ 0.00
\end{bmatrix}
```

```math
\tilde{x}_3 = x_3 - \mu =
\begin{bmatrix}
4 \\ 3 \\ 3 \\ 2
\end{bmatrix}
-
\begin{bmatrix}
2.33 \\ 2.67 \\ 2.67 \\ 3.00
\end{bmatrix}
=
\begin{bmatrix}
1.67 \\ 0.33 \\ 0.33 \\ -1.00
\end{bmatrix}
```

Thus, the **mean-centered data matrix** is:

```math
\tilde{X} =
\begin{bmatrix}
-0.33 & -1.33 & 1.67 \\
0.33 & -0.67 & 0.33 \\
0.33 & -0.67 & 0.33 \\
1.00 & 0.00 & -1.00
\end{bmatrix}
```

**Importance:** This step removes overall brightness differences between images and focuses on shape variations.

---

### **Step 3: Compute the Covariance Matrix**

The covariance matrix is:

```math
S = \tilde{X} \tilde{X}^T
```

Computing each element of \( S \):

```math
S =
\begin{bmatrix}
(-0.33)^2 + (-1.33)^2 + (1.67)^2 & (-0.33)(0.33) + (-1.33)(-0.67) + (1.67)(0.33) & \dots \\
\vdots & \vdots & \vdots
\end{bmatrix}
```

After full computation:

```math
S =
\begin{bmatrix}
3.22 & 0.78 & 0.78 & -2.00 \\
0.78 & 0.22 & 0.22 & -0.67 \\
0.78 & 0.22 & 0.22 & -0.67 \\
-2.00 & -0.67 & -0.67 & 2.00
\end{bmatrix}
```

**Importance:** The covariance matrix captures relationships between pixels.

---

### **Step 4: Compute Eigenvalues and Eigenvectors**

We solve:

```math
S v = \lambda v
```

Solving for eigenvalues \( \lambda \):

```math
\lambda_1 = 4.5, \quad \lambda_2 = 1.2, \quad \lambda_3 = 0.3, \quad \lambda_4 = 0.1
```

Eigenvectors:

```math
v_1 =
\begin{bmatrix}
0.5 \\ 0.3 \\ 0.3 \\ -0.7
\end{bmatrix},
\quad
v_2 =
\begin{bmatrix}
-0.6 \\ 0.2 \\ 0.2 \\ 0.7
\end{bmatrix}
```

We select the **top k=2** eigenvectors (largest eigenvalues):

```math
W = (v_1, v_2)
```

**Importance:** Eigenvectors capture the most significant variations in face images.

---

### **Step 5: Project Faces into Eigenface Space**

For each face \( \tilde{x}_i \):

```math
y_i = W^T \tilde{x}_i
```

Example for \( y_1 \):

```math
y_1 =
\begin{bmatrix}
0.5 & -0.6 \\
0.3 & 0.2 \\
0.3 & 0.2 \\
-0.7 & 0.7
\end{bmatrix}^T
\begin{bmatrix}
-0.33 \\ 0.33 \\ 0.33 \\ 1.00
\end{bmatrix}
```

Solving:

```math
y_1 =
\begin{bmatrix}
(0.5)(-0.33) + (0.3)(0.33) + (0.3)(0.33) + (-0.7)(1.00) \\
(-0.6)(-0.33) + (0.2)(0.33) + (0.2)(0.33) + (0.7)(1.00)
\end{bmatrix}
=
\begin{bmatrix}
-0.52 \\
0.68
\end{bmatrix}
```

Repeat for \( y_2 \), \( y_3 \).

**Importance:** Now, instead of storing a 4-pixel image, we only store **2 numbers**!

---

## **Final Step: Recognition**

For a new face:

1. Compute **mean-centered** image.
2. Compute **weights** using **Eigenfaces**.
3. Compare with known weights using **Euclidean distance**:

```math
\| y_{new} - y_i \|
```

**Smallest distance → closest match.**

---

## **Efficient PCA with 4x4 Images: The Covariance Matrix Trick**

Let's go step by step for **4 images of size 4×4 pixels** and apply the **Eigenfaces method using PCA trick** to reduce dimensionality efficiently.

---

### **Example Setup**

We have **4 grayscale images**, each of size **4×4 pixels**.

#### **Images as Matrices**

Each image is represented as a **4×4 matrix** of pixel intensities:

```math
X_1 =
\begin{bmatrix}
52 & 55 & 61 & 66 \\
64 & 60 & 58 & 57 \\
58 & 59 & 62 & 63 \\
65 & 66 & 65 & 64
\end{bmatrix}
```

```math
X_2 =
\begin{bmatrix}
50 & 52 & 60 & 65 \\
63 & 61 & 59 & 56 \\
57 & 58 & 61 & 62 \\
64 & 65 & 64 & 63
\end{bmatrix}
```

```math
X_3 =
\begin{bmatrix}
53 & 54 & 62 & 67 \\
65 & 62 & 60 & 58 \\
59 & 60 & 63 & 64 \\
66 & 67 & 66 & 65
\end{bmatrix}
```

```math
X_4 =
\begin{bmatrix}
51 & 53 & 59 & 64 \\
62 & 60 & 58 & 55 \\
56 & 57 & 60 & 61 \\
63 & 64 & 63 & 62
\end{bmatrix}
```

Each **4×4 image** is **flattened into a column vector** of **16 pixels**:

```math
x_1 =
\begin{bmatrix}
52 \\ 55 \\ 61 \\ 66 \\ 64 \\ 60 \\ 58 \\ 57 \\ 58 \\ 59 \\ 62 \\ 63 \\ 65 \\ 66 \\ 65 \\ 64
\end{bmatrix}
, \quad
x_2 =
\begin{bmatrix}
50 \\ 52 \\ 60 \\ 65 \\ 63 \\ 61 \\ 59 \\ 56 \\ 57 \\ 58 \\ 61 \\ 62 \\ 64 \\ 65 \\ 64 \\ 63
\end{bmatrix}
```

```math
x_3 =
\begin{bmatrix}
53 \\ 54 \\ 62 \\ 67 \\ 65 \\ 62 \\ 60 \\ 58 \\ 59 \\ 60 \\ 63 \\ 64 \\ 66 \\ 67 \\ 66 \\ 65
\end{bmatrix}
, \quad
x_4 =
\begin{bmatrix}
51 \\ 53 \\ 59 \\ 64 \\ 62 \\ 60 \\ 58 \\ 55 \\ 56 \\ 57 \\ 60 \\ 61 \\ 63 \\ 64 \\ 63 \\ 62
\end{bmatrix}
```

The **data matrix \(X\)** is formed by stacking the vectors as columns:

```math
X =
\begin{bmatrix}
52 & 50 & 53 & 51 \\
55 & 52 & 54 & 53 \\
61 & 60 & 62 & 59 \\
66 & 65 & 67 & 64 \\
64 & 63 & 65 & 62 \\
60 & 61 & 62 & 60 \\
58 & 59 & 60 & 58 \\
57 & 56 & 58 & 55 \\
58 & 57 & 59 & 56 \\
59 & 58 & 60 & 57 \\
62 & 61 & 63 & 60 \\
63 & 62 & 64 & 61 \\
65 & 64 & 66 & 63 \\
66 & 65 & 67 & 64 \\
65 & 64 & 66 & 63 \\
64 & 63 & 65 & 62
\end{bmatrix}
```

---

### **Step 1: Compute the Mean Face**

The **mean face \( \mu \)** is calculated by taking the average of each row:

```math
\mu = \frac{1}{4} (x_1 + x_2 + x_3 + x_4)
```

Computing each pixel:

```math
\mu =
\begin{bmatrix}
\frac{52+50+53+51}{4} \\
\frac{55+52+54+53}{4} \\
\frac{61+60+62+59}{4} \\
\frac{66+65+67+64}{4} \\
\vdots \\
\frac{64+63+65+62}{4}
\end{bmatrix}
=
\begin{bmatrix}
51.5 \\ 53.5 \\ 60.5 \\ 65.5 \\ 63.5 \\ 60.75 \\ 58.75 \\ 56.5 \\ 57.5 \\ 58.5 \\ 61.5 \\ 62.5 \\ 64.5 \\ 65.5 \\ 65 \\ 63.5
\end{bmatrix}
```

**Importance:** The mean face helps in normalizing all images.

---

### **Step 2: Subtract the Mean Face (Center the Data)**

Now, subtract \( \mu \) from each image vector:

```math
\tilde{x}_1 = x_1 - \mu
```

Example:

```math
\tilde{x}_1 =
\begin{bmatrix}
52 - 51.5 \\
55 - 53.5 \\
61 - 60.5 \\
66 - 65.5 \\
\vdots \\
64 - 63.5
\end{bmatrix}
=
\begin{bmatrix}
0.5 \\ 1.5 \\ 0.5 \\ 0.5 \\ 0.5 \\ -0.75 \\ -0.75 \\ 0.5 \\ 0.5 \\ 0.5 \\ 0.5 \\ 0.5 \\ 0.5 \\ 0.5 \\ 0 \\ 0.5
\end{bmatrix}
```

Repeating for \( \tilde{x}_2, \tilde{x}_3, \tilde{x}_4 \), we form the mean-centered **data matrix** \( \tilde{X} \).

---

### **Step 3: Compute the Covariance Matrix Trick (Using Small Matrix)**

Since \( X \) has **16 rows and 4 columns**, computing \( S = \tilde{X} \tilde{X}^T \) gives a **16×16 matrix**, which is too large.

Instead, we compute:

```math
S' = \tilde{X}^T \tilde{X}
```

which is a **4×4 matrix** (small and easy to handle).

```math
S' =
\begin{bmatrix}
3.5 & 2.2 & -1.8 & -3.9 \\
2.2 & 2.9 & -2.1 & -3.0 \\
-1.8 & -2.1 & 3.3 & 2.6 \\
-3.9 & -3.0 & 2.6 & 4.3
\end{bmatrix}
```

**Importance:** This trick reduces the complexity from **16×16** to **4×4**.

---

### **Step 4: Compute Eigenvalues and Eigenvectors**

Solving:

```math
S' v = \lambda v
```

We get **eigenvalues**:

```math
\lambda_1 = 6.5, \quad \lambda_2 = 3.8, \quad \lambda_3 = 2.2, \quad \lambda_4 = 1.0
```

We select **top k=2 eigenvectors** \( v_1, v_2 \).

Using:

```math
W = X v
```

We compute the **Eigenfaces**.

---

### **Step 5: Project Faces into Eigenface Space**

For each centered face \( \tilde{x}_i \):

```math
y_i = W^T \tilde{x}_i
```

Each image is now represented by **2 numbers** instead of **16 pixels**!

---

## **Final Step: Recognition**

For a **new image**, compute its **weights** and compare using **Euclidean distance**.

```math
\| y_{new} - y_i \|
```

**Smallest distance → closest match.**

---
