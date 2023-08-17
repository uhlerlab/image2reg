require("clusterProfiler")
require("ReactomePA")
require("ggplot2")
universe <- c('CDCA2', 'APPBP2', 'TK1', 'MMP2', 'PRKACA', 'CDC45', 'NOTCH3', 
              'SPARC', 'FBN1', 'FBLN1', 'RUNX1', 'SREBF1', 'STAC', 'TNNT1', 
              'GTSE1', 'TNNC1', 'YWHAQ', 'ERG', 'BIRC5', 'PRKCA', 'ACAN', 
              'CASP8', 'LZTS2', 'SPC25', 'MAPK7', 'S100A2', 'TUBB2B', 'CDC6', 
              'DHRS2', 'CEBPA', 'BEX1', 'UCHL5', 'HSPB1', 'WEE1', 'BMP4', 
              'PRAME', 'MYL2', 'SUV39H1', 'SDC2', 'TNC', 'DIABLO', 'DNMT3B', 
              'APC', 'TPM1', 'FKBP4', 'MAGEA6', 'LTBP2', 'CNN1', 'CLSPN', 
              'CRMP1', 'MCM4', 'GLI1', 'PKIA', 'CDKN1A', 'COL4A2', 'MAGEA1', 
              'MCM5', 'CDC42EP1', 'PCNA', 'JUN', 'NCF2', 'FOS', 'CD40', 
              'MAGED1', 'FBN2', 'MAP2K3', 'YES1', 'CDCA3', 'BCL2L11', 'AURKA',
              'CCND2', 'MAP3K5', 'PRC1', 'TP73', 'CYBA', 'CDK14', 'TIMP1', 
              'TLR4', 'MAGEA4', 'PLCG1', 'CEP192', 'FOXM1', 'HRAS', 'KRT18', 
              'NCAPH', 'UBE2S', 'FN1', 'KPNA2', 'PTN', 'MYH9', 'HJURP', 
              'MAP3K9', 'LGALS1', 'FGFR2', 'CDC42', 'SMAD4', 'UBE2C', 'E2F1', 
              'BAX', 'CRYAB', 'PDGFRB', 'OIP5', 'VCAN', 'COL5A1', 'MDC1', 
              'CCNE1', 'ELK1', 'ZWINT', 'FOXO1', 'SRC', 'PCOLCE', 'FGF1', 
              'PIK3R2', 'AMPH', 'SMO', 'DUSP6', 'LDOC1', 'RPS6KB1', 'TRIB3', 
              'TRAF2', 'KRAS', 'BRAF', 'KRT81', 'TRIP6', 'SKP2', 'NCAM1', 
              'TUBB', 'EPB41L3', 'MYBL2', 'IRAK4', 'NEK2', 'COL4A1', 'CREB1', 
              'KIF15', 'KRT80', 'ATXN7', 'CDK1', 'AKT1S1', 'RARA', 'CKS2', 
              'HSP90AB1', 'TGFB1', 'PRKCE', 'MAPK8', 'EFEMP1', 'MKI67', 
              'PTCH1', 'RELB', 'NDN', 'KRT8', 'RHOA', 'AURKB', 'PAK2', 
              'GSK3B', 'CLU', 'HAPLN1', 'STC2', 'TPX2', 'KLK6', 'SPC24', 
              'MCM7', 'CCNB1', 'MAGEA11', 'WWTR1', 'MBP', 'ALOX5', 'FEN1', 
              'FSTL1', 'HSPG2', 'VIM', 'MYD88', 'CDK6', 'TUBA1A', 'GLI2', 
              'NDC80', 'ARMCX2', 'CDC20', 'SDC1', 'MDFI', 'EEF1A2', 'DSP', 
              'FGFR3', 'IGFBP5', 'THY1', 'PLK1', 'PARP2', 'TONSL', 'TCF4', 
              'CCNB2', 'CENPE', 'HSPA1B', 'MFAP2', 'SFN', 'MYC', 'BUB1B', 
              'KIF4A', 'CCNF', 'GABRB3', 'MAP3K2', 'LAMA1', 'BUB1', 'LOXL4', 
              'C1QL1', 'NUF2', 'AXL', 'AQP1', 'SDC3', 'S100A14', 'LBH', 
              'TGM2', 'KIF23', 'INHBA', 'SMAD3', 'BCR', 'TNNT2', 'THBS1', 
              'TSC2', 'NEFL', 'RAF1', 'KIF2C', 'MCM10', 'KRT15', 'RRM2', 
              'TENM3', 'FHL2', 'MCM3', 'NCOR2', 'GNAQ', 'THRA', 'MYL9', 
              'CCNA2', 'IRF7', 'MSH2', 'COL1A1', 'EXO1', 'GIT1', 'PCLO', 
              'IGF2', 'TPM2', 'ATF4', 'CENPA', 'PLCB2', 'CDK2', 'TUBB6')

universe<-bitr(universe, fromType = "SYMBOL", toType = "ENTREZID", OrgDb="org.Hs.eg.db")$ENTREZID
universe <- sort(universe, decreasing=T)

cluster_list <- list("1"=c('ACAN', 'APPBP2', 'ATXN7', 'BMP4', 'COL1A1', 'COL4A1', 'COL4A2', 'COL5A1', 'DHRS2', 'FBLN1', 'FBN1', 'FBN2', 'FN1', 'FSTL1', 'HAPLN1', 'HSPG2', 'IGF2', 'IGFBP5', 'INHBA', 'KLK6', 'LAMA1', 'LRP1', 'LTBP2', 'MDK', 'MFAP2', 'MMP2', 'PCOLCE', 'PLAT', 'PTN', 'SDC1', 'SDC2', 'SDC3', 'SERPINE1', 'SPARC', 'STC2', 'TGFB1', 'TGM2', 'THBS1', 'THY1', 'TIMP1', 'TNC', 'VCAN'),
                     "2"=c('AMPH', 'AURKA', 'AURKB', 'BIRC5', 'BUB1', 'BUB1B', 'CCNF', 'CDCA2', 'CDCA8', 'CENPA', 'CENPE', 'CEP192', 'DNMT3B', 'GTSE1', 'HJURP', 'KIF15', 'KIF23', 'KIF2C', 'KIF4A', 'MKI67', 'NCAPH', 'NDC80', 'NEK2', 'NUF2', 'PRC1', 'PRSS23', 'PTCH1', 'RACGAP1', 'RRM2', 'SMO', 'SPC24', 'SPC25', 'TNNC1', 'TPX2', 'TTK', 'UBE2C', 'UBE2S', 'WEE1', 'ZWINT'),
                     "3"=c('AKT1S1', 'ALOX5', 'AQP1', 'ATF4', 'BAX', 'BCL2L11', 'CASP8', 'CD40', 'CDCA3', 'CEBPA', 'CLU', 'CREB1', 'CRYAB', 'DIABLO', 'EEF1A2', 'EFEMP1', 'ELK1', 'EPB41L3', 'ERG', 'FHL2', 'FOS', 'FOXO1', 'GABRB3', 'GLI1', 'GLI2', 'GNAQ', 'GSK3B', 'HSPA1B', 'IRAK4', 'IRF7', 'JUN', 'KPNA2', 'LBH', 'LOXL4', 'LZTS2', 'MAGEA1', 'MAGEA11', 'MAP2K3', 'MAP3K2', 'MAP3K5', 'MAPK7', 'MAPK8', 'MBP', 'MDC1', 'MYD88', 'NCOR2', 'NDN', 'PARP2', 'PLCB2', 'PRAME', 'PRKACA', 'PRKCA', 'RARA', 'RELB', 'RPS6KB1', 'RUNX1', 'SFN', 'SMAD4', 'SRC', 'SREBF1', 'STAC', 'SUV39H1', 'TCF4', 'TENM3', 'THRA', 'TLR4', 'TRAF2', 'TRIB3', 'TSC2', 'TUBB2B', 'WWTR1', 'YES1'),
                     "4"=c('ARMCX2', 'AXL', 'BCR', 'BEX1', 'BRAF', 'C1QL1', 'CDC42', 'CDC42EP1', 'CDK14', 'CNN1', 'CRMP1', 'CYBA', 'DSP', 'DUSP6', 'FGF1', 'FGFR2', 'FGFR3', 'FKBP4', 'GIT1', 'HK2', 'HRAS', 'HSP90AB1', 'HSPB1', 'KRT15', 'KRT18', 'KRT8', 'KRT80', 'KRT81', 'LDOC1', 'LGALS1', 'MAGEA12', 'MAGEA4', 'MAGEA6', 'MAGED1', 'MAP3K9', 'MDFI', 'MYH9', 'MYL2', 'MYL9', 'NCAM1', 'NCF2', 'NEFL', 'NOTCH3', 'PAK2', 'PCLO', 'PDGFRB', 'PIK3R2', 'PLCG1', 'PRKCE', 'RAF1', 'RHOA', 'S100A14', 'S100A2', 'TINAGL1', 'TNNT1', 'TNNT2', 'TPM1', 'TPM2', 'TUBA1A', 'TUBB', 'TUBB6', 'UCHL5', 'VIM'),
                     "5"=c('CCNA2', 'CCNB1', 'CCNB2', 'CCND2', 'CCNE1', 'CDC20', 'CDC45', 'CDC6', 'CDK1', 'CDK2', 'CDK6', 'CDKN1A', 'CKS2', 'CLSPN', 'E2F1', 'EXO1', 'FEN1', 'FOXM1', 'MCM10', 'MCM3', 'MCM4', 'MCM5', 'MCM7', 'MSH2', 'MYBL2', 'MYC', 'PCNA', 'PLK1', 'SKP2', 'TK1', 'TONSL', 'TP73', 'YWHAQ')
                     )
for(k in names(cluster_list)){
  cluster_list[[k]] <- bitr(cluster_list[[k]], fromType = "SYMBOL", toType = "ENTREZID", OrgDb="org.Hs.eg.db")$ENTREZID
}

compareGO <- compareCluster(geneClusters=cluster_list, fun="enrichGO", data="", OrgDb="org.Hs.eg.db", pAdjustMethod="BH", ont="BP", universe=universe)
dotplot(compareGO, font.size=10, showCategory=3,) + theme(
                                                       axis.text.y = element_text(face="bold", 
                                                                                  size=10))
comparePathway <- compareCluster(geneClusters=cluster_list, fun="enrichPathway", data="", organism="human", pAdjustMethod="BH", universe=universe)
dotplot(comparePathway)
compareKEGG <- compareCluster(geneClusters=cluster_list, fun="enrichKEGG", data="", organism="hsa", pAdjustMethod="BH", universe=universe)
dotplot(compareKEGG)
