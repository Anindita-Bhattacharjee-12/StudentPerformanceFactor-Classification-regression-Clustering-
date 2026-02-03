# ============================================
# Project - Option 1 (Regression)
# Dataset : Student Performance Factors
# Source  : Kaggle (loaded via Google Drive URL)
# Link    : https://drive.google.com/uc?export=download&id=1IWA5bJ8t7s0pJKCGah4Fg4xVEFWS1SD5
# Domain  : Education
# Objective: Predict continuous Exam_Score using Multiple Linear Regression
# Notes:
# - Structured dataset with 6,607 rows and 20 columns
# - Mix of numerical features (e.g., Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, Exam_Score)
#   and categorical features (e.g., Parental_Involvement, School_Type, Gender, Motivation_Level)
# ============================================

# ===============================
# STEP 1: Load Required Packages
# ===============================

packages <- c(
  "ggplot2","dplyr","tidyr","fastDummies","corrplot"
)

for(pkg in packages){
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# ===============================
# STEP 2: Load Dataset
# ===============================

url <- "https://drive.google.com/uc?export=download&id=1IWA5bJ8t7s0pJKCGah4Fg4xVEFWS1SD5"
data <- read.csv(url, stringsAsFactors = FALSE)

# Convert empty strings ("") to NA (consistent with classification script)
data[data == ""] <- NA

cat("Dataset loaded successfully\n")
cat("Rows:", nrow(data), "Columns:", ncol(data), "\n")
cat("Names:\n")
print(names(data))
cat("Summary:\n")
print(summary(data))
cat("NA counts:\n")
print(colSums(is.na(data)))

# ===============================
# STEP 3: Data Understanding
# ===============================

dim(data)
str(data)
summary(data)

# Identify numeric and categorical features
num_cols <- sapply(data, is.numeric)
cat_cols <- sapply(data, is.character)

cat("\nNumeric Features:\n")
print(names(data)[num_cols])

cat("\nCategorical Features:\n")
print(names(data)[cat_cols])

# ===============================
# STEP 3.1: Identify Target Variable
# ===============================

target <- colnames(data)[ncol(data)]
cat("Target variable:", target, "\n")

# ===============================
# STEP 4: Exploratory Data Analysis 
# ===============================

# 4.1 Numeric vs Target
num_cols[target] <- FALSE
num_predictors <- names(data)[num_cols]

# Histogram of target
ggplot(data, aes(.data[[target]])) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  theme_minimal() +
  ggtitle(paste("Distribution of", target))
cat("Histogram shows the distribution of the target variable.\n")

# Boxplot of target
ggplot(data, aes(y = .data[[target]])) +
  geom_boxplot(fill = "lightgreen") +
  theme_minimal() +
  ggtitle(paste("Boxplot of", target))
cat("Boxplot shows the spread and potential outliers of the target variable.\n")

# Scatter plots of numeric predictors vs target
for(col in num_predictors){
  p <- ggplot(data, aes(x = .data[[col]], y = .data[[target]])) +
    geom_point(alpha = 0.5, color = "darkblue") +
    geom_smooth(method = "lm", se = FALSE, color = "red") +
    theme_minimal() +
    ggtitle(paste(target, "vs", col))
  print(p)
  cat(paste("Scatter plot shows the relationship between", col, "and", target, "\n"))
}

# Correlation matrix
if (length(num_predictors) > 1) {
  cor_matrix <- cor(data[, num_predictors], use = "pairwise.complete.obs")
  corrplot(cor_matrix, method = "color", type = "upper",
           tl.cex = 0.6, title = "Correlation Matrix (Numeric Features)",
           mar = c(0,0,1,0))
  cat("Correlation matrix shows how numeric predictors are correlated with each other.\n")
}

# Categorical variables
cat_cols[target] <- FALSE
cat_predictors <- names(data)[cat_cols]

# Bar plots for categorical variables
for(col in cat_predictors){
  p <- ggplot(data, aes(x = .data[[col]])) +
    geom_bar(fill = "orange") +
    theme_minimal() +
    ggtitle(paste("Count plot of", col))
  print(p)
  cat(paste("Bar plot shows the frequency distribution of", col, "\n"))
}

# ===============================
# STEP 5: Data Preprocessing
# ===============================

# 5.1 Missing Values
na_count <- sapply(data, function(x) sum(is.na(x)))
cat("Missing values per column:\n")
print(na_count)

# Numeric → median
for(col in names(data)[num_cols]){
  data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
}

# Categorical → mode
get_mode <- function(x){
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
for(col in names(data)[cat_cols]){
  data[[col]][is.na(data[[col]])] <- get_mode(data[[col]])
}

# Check NA after preprocessing
cat("\nNA counts after missing value imputation:\n")
print(colSums(is.na(data)))

# 5.2 Remove Duplicate Rows
dup_count <- sum(duplicated(data))
cat("Duplicate rows found:", dup_count, "\n")
data <- data[!duplicated(data), ]
cat("Dataset size after removing duplicates:", dim(data), "\n")

# ===============================
# 5.3 Handle Outliers (IQR capping)
# ===============================
numerical_vars <- names(data)[sapply(data, is.numeric)]
numerical_vars <- setdiff(numerical_vars, target)  # exclude target

cap_outliers_iqr <- function(x){
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5*IQR
  upper <- Q3 + 1.5*IQR
  x[x < lower] <- lower
  x[x > upper] <- upper
  return(x)
}

for (var in numerical_vars) {
  data[[var]] <- cap_outliers_iqr(data[[var]])
  cat(var, ": Outliers capped using IQR method\n")
}

# ===============================
# 5.4 Encode Categorical Variables
# ===============================
data <- fastDummies::dummy_cols(
  data,
  select_columns = cat_predictors,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

# ===============================
# 5.5 Feature Engineering
# ===============================
# Apply transformations to reduce skewness and capture non-linear effects
if("Previous_Scores" %in% names(data)){
  data$log_Previous_Scores <- log(data$Previous_Scores + 1)
}
if(all(c("Hours_Studied","Attendance") %in% names(data))){
  data$Hours_Attendance <- data$Hours_Studied * data$Attendance
}

predictors <- setdiff(names(data), target)

# ===============================
# 5.6 Feature Scaling
# ===============================
data[, predictors] <- scale(data[, predictors])

# Final check for NA before modeling
cat("\nNA counts after full preprocessing:\n")
print(colSums(is.na(data)))

# ===============================
# STEP 6: Train-Test Split
# ===============================
set.seed(123)
index <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))
train <- data[index, ]
test  <- data[-index, ]

cat("Train size:", nrow(train), " Test size:", nrow(test), "\n")

# ===============================
# STEP 7: Regression Modeling & Evaluation
# ===============================
model <- lm(as.formula(paste(target, "~ .")), data = train)
summary(model)

# Predictions
pred   <- predict(model, test)
actual <- test[[target]]

# Metrics
RMSE <- sqrt(mean((pred - actual)^2))
MAE  <- mean(abs(pred - actual))
R2   <- cor(pred, actual)^2

cat("RMSE        :", RMSE, "\n")
cat("MAE         :", MAE, "\n")
cat("R-squared   :", R2, "\n")
cat("Adjusted R2 :", summary(model)$adj.r.squared, "\n")

cat("\n=== Regression Model Interpretation ===\n")
cat("R-squared of", round(R2, 3), "indicates that the model explains",
    round(R2 * 100, 1), "% of the variance in", target, ".\n")
cat("RMSE of", round(RMSE, 3),
    "shows the average prediction error in the same units as", target, ".\n")
cat("Lower MAE and RMSE and higher R-squared indicate better predictive performance.\n")

# ===============================
# STEP 8: Feature Importance
# ===============================
coef_df <- data.frame(
  Feature = names(coef(model))[-1],
  Coefficient = coef(model)[-1]
) %>% arrange(desc(abs(Coefficient)))

cat("Top 10 features by absolute coefficient:\n")
print(head(coef_df, 10))

# ===============================
# STEP 9: Actual vs Predicted Plot
# ===============================
ggplot(data.frame(Actual = actual, Predicted = pred),
       aes(Actual, Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "red") +
  theme_minimal() +
  ggtitle("Actual vs Predicted Values")
cat("Scatter plot shows how well predicted values match actual values.\n")

# ===============================
# STEP 10: Sample Prediction (5 Rows)
# ===============================
sample_test <- head(test, 5)
sample_pred <- predict(model, sample_test)
sample_actual <- sample_test[[target]]

sample_results <- data.frame(
  Student_ID = rownames(sample_test),
  Actual = sample_actual,
  Predicted = sample_pred
)

cat("Sample predictions for 5 students:\n")
print(sample_results)

ggplot(sample_results, aes(x = Actual, y = Predicted, label = Student_ID)) +
  geom_point(color = "purple", size = 3) +
  geom_text(nudge_y = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  ggtitle("Actual vs Predicted for 5 Sample Students")
cat("Scatter plot shows predictions vs actual for 5 sample students.\n")
