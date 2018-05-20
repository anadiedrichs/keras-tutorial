#' AUXILIAR FUNCTIONS
#' metric_mean_absolute_error(y_true, y_pred)
mean_absolute_error <- function(y_true,y_pred){
  return(sum(abs(y_true-y_pred))/length(y_true))
}
rmse <- function(y_true,y_pred){
  return(sqrt(sum((y_true-y_pred)^2)/length(y_true)))
}
#' performance evaluation (errors in testset)
evaluation <- function(model, samples, target, namePlot)
{
  pred <- as.vector(predict_on_batch(model,samples))
  # pasamos la info a grados Celsius
  pred.g <- pred * std[predictor.target]
  target.g <- targets * std[predictor.target]
  #' Calculamos MAE y RMSE
  #' 
  mae <- mean_absolute_error(target.g, pred.g)
  rmserror <- rmse(target.g, pred.g)
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
  t= data.frame(unclass(table(real.cut,pred.cut)))
  write.csv(t,file=paste(namePlot,"-cm.csv",sep="")) #PROBAR
  write.csv(cbind(c("MAE: ",mae),c("RMSE:",rmserror)),file = paste(namePlot,"-err.csv",sep="")) #PROBAR
  write.csv(cbind(pred.g,target.g),file = paste(namePlot,"-realVsPred.csv",sep="") ) #PROBAR
  
  return(c(mae = mae,rmse = rmserror, real = target.g, pred=pred.g))
  
}