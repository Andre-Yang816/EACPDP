library(ScottKnottESD)
finalpath= "../target/KNN/"
skresultpath='../output/KNN/'
file_names<- list.files(finalpath)
for (i in 1:length(file_names)) {
    
    path=paste(finalpath,sep = "",file_names[i])
    print(path)
    csv<- read.table(file=path, header=TRUE, sep=",")
    csv<-csv[-1]
    sk <- sk_esd(csv)
    #plot(sk)
    
    resultpath=paste(skresultpath,sep = "",file_names[i])
    resultpath=paste(resultpath,sep = "",".txt")
    print(resultpath)
    
    write.table (sk[["groups"]], resultpath) 
}
