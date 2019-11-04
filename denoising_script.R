#Data
library(readr)
library(BBmisc)
final <- read_csv("C:/Users/claba/OneDrive/Desktop/final.csv")

# Encoding
# Claudio: 1
# Egon: 2
# Lorenzo: 3
# Casa: 3
# Metro: 1
# Strada: 2


# Target encoding
final$guy <- as.numeric(as.factor(final$guy))
lab_guy_full <- final$guy
final$location <- as.numeric(as.factor (final$location))
lab_location_full <- final$location


number_of_observations <- dim(final)[2]
number_of_coefficients <- dim(final)[2]-4

# Matrix of functions with target building and normalization
functions <- as.matrix(final[,5:number_of_observations])
functions <- cbind(normalize(functions, range = c(0,10)),lab_guy_full,lab_location_full)
functions <- functions[which(final["sensor"] == "DecibelSource" | final["sensor"] == "PitchSensor"),]
number_of_final_functions <- dim(functions)[1]/2

# Matrix of averaged functions with target building 
D <- matrix(NA, nrow = number_of_final_functions, ncol = number_of_coefficients+2)
j = 1
for (i in seq(1,(number_of_final_functions*2),2)){
  D[j,] = ((functions[i,]+functions[i+1,])/2)
  j = j+1
}
lab_guy <- D[,dim(D)[2]-1]
lab_location <- D[,dim(D)[2]]
D <- D[,1:(dim(D)[2]-2)] 

# Basis mat
cos.basis <- function(x = (1:number_of_coefficients)/number_of_coefficients, j.max = length(x)){
  n = length(x)
  mat.basis <- matrix(NA, nrow = n, ncol = j.max)
  mat.basis[,1] <- 1
  for (j in 2:j.max){
    mat.basis[,j] = sqrt(2)*cos((j-1)*pi*x)
  }
  return(mat.basis)
}

mat <- t(cos.basis())

# Get coeff
Z_ <- (1/number_of_coefficients)*t(mat %*% t(D))


# HF variance
HF.var <- function(Z, j.cut = round(length(Z)/4)){
  n = length(Z)
  min.idx = n - (j.cut - 1)
  out = (n/j.cut)*sum( (Z[min.idx:n])^2 )
  return(out)
}

# Risk estimator related functions
risk_est <- function(J,Z_row,sigma2 = HF.var(Z_row)){#?.5
  m <- length(Z_row)
  return(J*(sigma2/m) + sum((Z_row[(J+1):m]^2-sigma2/m)))#*((Z_row[(J+1):m]^2-sigma2/m) > 0)? MANNAGGIA LA MAIALA
}

total_regret <- function(J_,Z){
  R <-  apply(Z,1,risk_est, J = J_)
  return(sum(R-min(R)))
} 

#J choice
J_range <- 1:(number_of_coefficients-1)
res <- sapply(J_range, total_regret, Z = Z_)
J_hat <- J_range[which(res == min(res))]

# Flag to work on places or guys
guys = 0


if (guys){

# Coeff cutting guys
Z_cut <- Z_[,1:J_hat]

# Building of final coefficients matrix and data splitting
Z_labeled <- cbind(Z_cut,lab_guy)
set.seed(7)
ss <- sample(1:2,size=dim(D)[1],replace=TRUE,prob=c(0.8,0.2))
train <- Z_labeled[ss==1,]
test <- Z_labeled[ss==2,]

# Saving
write.csv(train,file = "train.csv")
write.table(test,file = "test.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")

# Just a look
denoised_funct <- t(t(mat)[,1:J_hat] %*% t(Z_cut))
plot(D[2,], type = "l")
plot(denoised_funct[2,],type = "l")
}
if(guys == 0){

  # Coeff cutting locus amoenus
  j_cut = round(number_of_coefficients/4)
  min_idx = number_of_coefficients - (j_cut - 1)
  Z_cut <- Z_[,min_idx:number_of_coefficients]
  
  # Building of final coefficients matrix and data splitting
  Z_labeled <- cbind(Z_cut,lab_location)
  set.seed(7)
  ss <- sample(1:2,size=dim(D)[1],replace=TRUE,prob=c(0.8,0.2))
  train <- Z_labeled[ss==1,]
  test <- Z_labeled[ss==2,]
  
  # Saving
  write.csv(train,file = "train.csv")
  write.table(test,file = "test.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
}


