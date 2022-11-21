/*
 * testeVideoModulo02Lib.h
 *
 *      Author: gustavo
 */

#ifndef TESTEVIDEOMODULO02LIB_H_
#define TESTEVIDEOMODULO02LIB_H_






#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "math.h"
#include <stdlib.h>
#include <time.h>

using namespace cv;

struct structAngulo
{
	std::vector<Point2f> angulo;
	double media;
	double desvioPadrao;
	int tamanhoVetor;
	structAngulo* anterior;
	structAngulo* posterior;
};

void readMe();

void imprimeMatriz(Mat& matriz);

void detect_feature_detector(int siftOrSurf, int minHessian, Mat& img_reference, std::vector<KeyPoint>& keypoints_reference, Mat& img_target, std::vector<KeyPoint>& keypoints_target);

void detect_descriptor_extractor(int siftOrSurf, Mat& img_reference, std::vector<KeyPoint>& keypoints_reference, Mat& descriptors_reference, Mat& img_target, std::vector<KeyPoint>& keypoints_target, Mat& descriptors_target);

void removePontosRepetidos(std::vector<Point2f>& reference, std::vector<Point2f>& target, std::vector<Point2f>& reference_corrected, std::vector<Point2f>& target_corrected);

void transformaVectorEmMat(std::vector<Point2f>& vetor, Mat& matriz);

void moveCentroideOrigem(Mat& originalNorm, double& subtrair_eixo_x, double& subtrair_eixo_y);

double isotropicScaling(Mat& originalNorm);

void normalizando(Mat& original, Mat& originalNorm, Mat& T);

int eInversivel(Mat& T);

void coorHomogeneas(Mat& MatNorm, Mat& MatNormHom);

void generateRandomVec(Mat& PnX1, Mat& PnX2, int n, int tamanho, Mat& Ref, Mat& Tar);

int verifyCollinearity(Mat& PnX1, int n);

void preencheMatrizA(Mat& A, int& ptHfit1, int& ptHfit2, Mat& PnX1Norm, Mat& PnX2Norm);

int HFitting(int auxHfit, Mat& PnX1, Mat& PnX2, int& degenerate, Mat& Htemp);

void criaHX1(Mat& HX1, Mat& Htemp, Mat& referenceNormHom);

void criaInvHX2(Mat& invHX2, Mat& Htemp, Mat& targetNormHom);

void normalizaPelaTerceiraColuna(Mat& H, Mat& HNorm);

int calculaD2(Mat& d2, Mat& HX1Norm, Mat& invHX2Norm, Mat& referenceNormHom, Mat& targetNormHom);

int countingInliers(Mat& d2, double t);

void preencheInliers(Mat& inliersX1, Mat& inliersX2, Mat& inliers, Mat& referenceNormHom, Mat& targetNormHom, Mat& d2, double t);

void calculaMatrizCovariancia(Mat& inliers, Mat& covar);

void calculaMatrizMaiorCovariancia(Mat& inliers, Mat& covar);

double calculaScatter(std::vector<double>& eigenvalues);

void geraImagemDupla(Mat& img_reference, Mat& img_target, Mat& imgDuplaColor);

void desnormaliza(Mat& Tinv, Mat& bestInliersX, std::vector<Point2f>& bestInXVec);

void desenhando(Mat& imgDupla, std::vector<Point2f>& bestInX1Vec, std::vector<Point2f>& bestInX2Vec);

void desenhaPontos(Mat& imgDupla, Mat& T1, Mat& T2, Mat& bestInliersX1, Mat& bestInliersX2);







int modifiedHomography(Mat& Hinter, std::vector<Point2f>& reference, std::vector<Point2f>& target, Mat& imgDupla);

int calculaNCCImage(Mat& warpImg, Mat& img_target, Mat& NCC_img_8b, int dimensao, Mat& warpImgMask);

void geraInicioFilaHistogramaAngulo(std::vector<Point2f>& reference_corrected, std::vector<Point2f>& target_corrected, structAngulo& inicial, double& dist);

void imprimeElementoHistogramaAngulo(structAngulo& inicial);

void geraFilaHistogramaAngulo(int quantidade, structAngulo& inicial, structAngulo* inicioFila);

void removeAnguloGrande(std::vector<Point2f>& reference_corrected, std::vector<Point2f>& target_corrected, std::vector<Point2f>& reference_angulo, std::vector<Point2f>& target_angulo, double& controlaIntervaloAngulo, double& dist);

void removeDistanciaDispar(std::vector<Point2f>& reference_angulo, std::vector<Point2f>& target_angulo, std::vector<Point2f>& reference_distancia, std::vector<Point2f>& target_distancia, double& dist, double& intervaloTamanho, int& soEixoX);

int calculaNCCImagePosAbsDiff(Mat& warpImg, Mat& img_target, Mat& NCC_img_8b_porAbsDiff, Mat& abs_imgs_diff, int dimensao, double limiarAbsDiff, Mat& warpImgMask);

void criaMascara(Mat& NCC_img_8b, Mat& Mask, double& threshold, Mat& warpImgMask);

void criaWarpMask(Mat& warpImgMask, Mat& H);

void corrigeAbsDiffImg(Mat& abs_imgs_diff, Mat& abs_imgs_diff_warpMask, Mat& warpImgMask);

void corrigeAbsDiffImgdouble(Mat abs_imgs_diff, Mat& abs_imgs_diff_warpMask, Mat warpImgMask);

void calculaMaxS(double& maxS, int tamVecMaxMinS, Mat& matS);

void calculaMinS(double& minS, int tamVecMaxMinS, Mat& matS);

void calculaCS(Mat& matS, int tamVecMaxMinS, double& CS, double comparaCS);

int geraBitimg(Mat& img, int tamJanelaS, double comparaCS, int tamVecMaxMinS, double somaI, Mat& bitMat);

void geraBintra(Mat& Bintra, Mat& abs_imgs_diff, double threshold3);

void calculaMatIntersecao(Mat& intersecao, Mat& MatA, Mat& MatB);

void calculaMatSubtracao(Mat& subtracao, Mat& MatA, Mat& MatB);

void adaptaBintra(Mat& Bintra, Mat& BintraAdapt);

void geraB3(Mat& B3, Mat& B2, Mat& BintraTar, Mat& BintraRef);



void geraVetoresEnviarCalcularHomografia(int contadorMax, double distMult, std::vector< DMatch >& good_matches, double intervaloTamanho, int soEixoX, double controlaIntervaloAngulo, unsigned int numMinimoPares, int usaTotal, int siftOrSurf, int minHessian, Mat& img_reference, std::vector<KeyPoint>& keypoints_reference, Mat& descriptors_reference, Mat& img_target, std::vector<KeyPoint>& keypoints_target, Mat& descriptors_target, std::vector<Point2f>& reference_enviar, std::vector<Point2f>& target_enviar);

int calculaHomografia(int qualRansac, Mat& H, std::vector<Point2f>& reference_enviar, std::vector<Point2f>& target_enviar, double& multHomographyOpenCV, Mat& imgDupla);

void desenhaHomografiaImgMatches(Mat& img_matches, int numColunas, int numLinhas, Mat& H);

void geraB1(Mat& B1, Mat& warpImg, Mat& H, int usaAbsDiff, Mat& img_target, int dimensaoJanela, double limiarAbsDiff, double threshold);

void corrigeHinterWarpedIntraRefMask(Mat& HinterWarpedIntraRefMask, Mat& HinterWarpedIntraRefMaskCorrigida, Mat& Hinter);

void geraB2(Mat& B2, Mat& HinterWarpedIntraRefMaskCorrigida, Mat& B1);

void gerandoBintra(Mat& Bintra, double threshold, Mat& warpImg, Mat& img_target, int tamJanelaS, double comparaCS, int tamVecMaxMinS, double somaI, Mat& MaskAux);

void corrigeB3tempMask(Mat& matAux, Mat& matAux_warpMask, Mat& warpMask);

void corrigeVotingMask(Mat& matAux, Mat& matAuxcorrected, Mat& mask);

unsigned int descobreQuadroMudancaDirecao(Mat& vetorAVarrer, double coefAngTemplate01, double coefLinearTemplate01, double coefAngTemplate02, double coefLinearTemplate02, unsigned int margemVetorAVarrer, unsigned int tamVetorAVarrer, double pontoIntersecaoX);

void qualOSentidoInicialDoMovimento(unsigned int& numFramesVerificaSentidoInicialAux, unsigned int numFramesVerificaSentidoInicial, double& verificaSentidoInicial, double deslocamentoHorizontalHomografia, int& sentidoVideoRef, unsigned int& verificaoSentidoInicialRealizada);

int detectaPrimeiraMudancaDeSentido(double deslocamentoHorizontalHomografia, double& verificaSentidoInicial, int& sentidoVideoRef);

int detectaDemaisMudancasDeSentido(int& varAuxImprimindo, unsigned int& emQualTroca, unsigned int& contaFrames, double& posEixoY, double deslocamentoHorizontalHomografia, unsigned int comecaAGerarVetorAVarrer, unsigned int tamVetorAVarrer, Mat& vetorAVarrer, unsigned int& contadorAuxiliarVetor, int& sentidoVideoRef, unsigned int& frameDaMudancaDeDirecao, unsigned int margemVetorAVarrer, double coefAngTemplate01EsqDir, double coefLinearTemplate01EsqDir, double coefAngTemplate02EsqDir, double coefLinearTemplate02EsqDir, double pontoIntersecaoXEsqDir, double coefAngTemplate01DirEsq, double coefLinearTemplate01DirEsq, double coefAngTemplate02DirEsq, double coefLinearTemplate02DirEsq, double pontoIntersecaoXDirEsq, cv::VideoCapture& vcapRef, const std::string videoStreamAddressRefDirEsq, const std::string videoStreamAddressRefEsqDir, Mat& imageRefColor);

void preencheCorBranca(Mat& preencheAux);

int calcula_quantidade_manchas(Mat& matOriginal, int threshold_multirresolucao);

void gera_mat_detect_final_multirresolucao(Mat& fusao_mascaras_voting, std::vector<Mat>& count_mask_vec, double delta_janela, std::vector<double> multiplica_area_janela_vec, std::vector<int> dimensaoJanelaNCC_vec, std::vector<unsigned int> subamostraPorNCC_vec, unsigned int num_multres, int threshold_multirresolucao, Mat img_target_big, unsigned int usaRectouAreaDaMancha, unsigned int divisorShow);


void removeDiferentScale(const std::vector<KeyPoint>& keypoints_object_in, const std::vector<KeyPoint>& keypoints_scene_in, const std::vector< DMatch > matches, std::vector< DMatch >& matches_out);


void retrievePoint2fFromMatches(const std::vector<KeyPoint>& keypoints_object_in, const std::vector<KeyPoint>& keypoints_scene_in, const std::vector< DMatch > matches, std::vector<Point2f>& keypoints_object_out, std::vector<Point2f>& keypoints_scene_out);


#endif /* TESTEVIDEOMODULO02LIB_H_ */
