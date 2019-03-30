#' AUXILIAR FUNCTIONS
#' metric_mean_absolute_error(y_true, y_pred)
library(caret)
RESULT_FILE <- "result-2018-07-16.csv"
#R squared

rsq <- function(x, y){ summary(lm(y~x))$r.squared } #tested

# Function that returns Mean Absolute Error
mean_absolute_error <- function(y_true,y_pred){
  mean(abs(y_true-y_pred))
}

# Function that returns Root Mean Squared Error
rmse <- function(y_true,y_pred){
  sqrt(mean((y_true-y_pred)^2))
}

#' performance evaluation (errors in testset)
evaluation <- function(model, samples, target, namePlot)
{
  pred <- as.vector(predict(model,samples))
  # pasamos la info a grados Celsius
  pred.g <- (pred * std[predictor.target]) + mean[predictor.target]
  target.g <- (target * std[predictor.target])  + mean[predictor.target]
  #' Calculamos MAE y RMSE
  #' 
  mae <- mean_absolute_error(target.g, pred.g)
  rmserror <- rmse(target.g, pred.g)
  r2 <- rsq(target.g, pred.g)
  #' Graficar
  #' 
  resultados <- data.frame(index=1:length(pred.g),real=target.g,pred=pred.g)
  ggplot(resultados, aes(index)) + 
    geom_line(aes(y = real, colour = "real")) + 
    geom_line(aes(y = pred, colour = "pred"))
  ggsave(paste(namePlot,".png",sep="")) # PROBAR ESTO
  #' TODO save plot
  #' 
  #' Caso heladas vs no heladas + matriz de confusión
  #' 
  real.cut <- cut(target.g,breaks = c(-20,0,20))
  pred.cut <- cut(pred.g,breaks = c(-20,0,20))
  c <- confusionMatrix(pred.cut,real.cut) # pasar pred and then truth or real values
  #write.csv(t,file=paste(namePlot,"-cm.csv",sep="")) #PROBAR
  #write.csv(cbind(c("MAE: ",mae),c("RMSE:",rmserror)),file = paste(namePlot,"-err.csv",sep="")) #PROBAR
  write.csv(cbind(pred.g,target.g),file = paste(namePlot,"-realVsPred.csv",sep="") ) #PROBAR
  res <- cbind(namePlot,mae,r2,rmserror,c$byClass["Sensitivity"],c$byClass["Precision"],c$table[1,1],c$table[1,2],c$table[2,2],c$table[2,1])
  #write.csv(data.frame("dataset-config","mae","r2","rmse","recall","precision","TP","FP","TN","FN"),RESULT_FILE)
  write.table(res, RESULT_FILE, sep = ",", col.names = F, append = T)
  return(c(mae = mae,rmse = rmserror, real = target.g, pred=pred.g))
  
}
