Musson_y_Allen = function(c1,c2,c3,M,R){
    .expr_c1 = c1
    .expr_c2 = c2 * M
    .expr_c3 = c3 * log(R)
    .value = .expr_c1 + .expr_c2 + .expr_c3
    .grad <- array(0, c(length(.value), 3L), list(NULL, c("c1","c2","c3")))
    .grad[, "c1"] <- 1.0
    .grad[, "c2"] <- M
    .grad[, "c3"] <- log(R)
    attr(.value, "gradient") <- .grad
    .value  
}

Musson_y_Allen_noM = function(c1,c2,R){
    .expr_c1 = c1
    .expr_c2 = c2 * log(R)

    .value = .expr_c1 + .expr_c2
    .grad <- array(0, c(length(.value), 2L), list(NULL, c("c1","c2")))
    .grad[, "c1"] <- 1.0
    .grad[, "c2"] <- log(R)
    attr(.value, "gradient") <- .grad
    .value  
}

Musson_y_Allen_noM_f0 = function(c1,c2,D,f0,R){
    g_f0 <- ifelse(f0 <= 1, log10(f0), 0)
    .expr_c1 = c1
    .expr_c2 = c2 * log(R)
    .expr_D = D * g_f0

    .value = .expr_c1 + .expr_c2 + .expr_D
    .grad <- array(0, c(length(.value), 3L), list(NULL, c("c1","c2","D")))
    .grad[, "c1"] <- 1.0
    .grad[, "c2"] <- log(R)
    .grad[, "D"] <- g_f0
    attr(.value, "gradient") <- .grad
    .value  
}

Bakun = function(c1,c2,c3,c4,M,R){
    .expr_c1 = c1
    .expr_c2 = c2 * M
    .expr_c3 = c3 * log(R)
    .expr_c4 = c4 * R
    .value = .expr_c1 + .expr_c2 + .expr_c3 + .expr_c4
    .grad <- array(0, c(length(.value), 4L), list(NULL, c("c1","c2","c3","c4")))
    .grad[, "c1"] <- 1.0
    .grad[, "c2"] <- M
    .grad[, "c3"] <- log(R)
    .grad[, "c4"] <- R
    attr(.value, "gradient") <- .grad
    .value  
}

Bakun_noM = function(A,B,C,R){
    .expr_A = A    
    .expr_B = B * log(R)
    .expr_C = C * R
    .value = .expr_A + .expr_B + .expr_C
    .grad <- array(0, c(length(.value), 3L), list(NULL, c("A","B","C")))
    .grad[, "A"] <- 1.0
    .grad[, "B"] <- log(R)
    .grad[, "C"] <- R
    attr(.value, "gradient") <- .grad
    .value  
}

Bakun_noM_f0 = function(A,B,C,D,R,f0){
   # g(f0) = log10(f0) si f0 <= 1, sino 0
    g_f0 <- ifelse(f0 <= 1, log10(f0), 0)
    .expr_A = A    
    .expr_B = B * log(R)
    .expr_C = C * R
    .expr_D = D * g_f0
    .value = .expr_A + .expr_B + .expr_C + .expr_D
    .grad <- array(0, c(length(.value), 4L), list(NULL, c("A","B","C","D")))
    .grad[, "A"] <- 1.0
    .grad[, "B"] <- log(R)
    .grad[, "C"] <- R
    .grad[, "D"] <- g_f0
    attr(.value, "gradient") <- .grad
    .value  
}



Atkinson = function(c1,c2,c3,c4,c5,M,R){
    .expr_c1 = c1
    .expr_c2 = c2 * M
    .expr_c3 = c3 * log(R)
    .expr_c4 = c4 * R
    .expr_c5 = c5 * M * log(R)
    .value = .expr_c1 + .expr_c2 + .expr_c3 + .expr_c4 + .expr_c5
    .grad <- array(0, c(length(.value), 5L), list(NULL, c("c1","c2","c3","c4","c5")))
    .grad[, "c1"] <- 1.0
    .grad[, "c2"] <- M
    .grad[, "c3"] <- log(R)
    .grad[, "c4"] <- R
    .grad[, "c5"] <- M * log(R)
    attr(.value, "gradient") <- .grad
    .value  
}

nehrp_class <- function(vs30) {
  ifelse(
    is.na(vs30), NA,
    ifelse(
      vs30 >= 1500, "A",
      ifelse(
        vs30 >= 760, "B",
        ifelse(
          vs30 >= 360, "C",
          ifelse(
            vs30 >= 180, "D",
            "E"
          )
        )
      )
    )
  )
}