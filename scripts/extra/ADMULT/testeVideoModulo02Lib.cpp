/*
 * testeVideoModulo02Lib.cpp
 *
 *      Author: gustavo
 */

#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// #include <opencv2/nonfree/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv/cxcore.h"
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core_c.h>
#include "math.h"
#include <stdlib.h>
#include <time.h>
#include "testeVideoModulo02Lib.h"

using namespace cv;
using namespace std;

/*struct structAngulo
{
	std::vector<Point2f> angulo;
	double media;
	double desvioPadrao;
	int tamanhoVetor;
	structAngulo *anterior;
	structAngulo *posterior;
};*/

void readMe()
{
	std::cout << " Usage: ./detect <img_reference> <img_target> <multiplicador_SIFT/SURF> <escolher SIFT (qualquer numero) ou SURF (0)> <minHessian (ex: 400)> <ransacReprojThreshold_opencv (ex: 2.0)> <qualRansac (0 para o do OpenCV, e qualquer outro valor para o modificado)>" << std::endl;
}

void imprimeMatriz(Mat &matriz)
{
	for (int k = 0; k < matriz.rows; k++)
	{
		for (int l = 0; l < matriz.cols; l++)
		{
			printf("%5.20f ", matriz.at<double>(k, l));
		}
		printf("\n");
		fflush(stdout);
	}
	printf("\n");
	fflush(stdout);
}

void detect_feature_detector(int siftOrSurf, int minHessian, Mat &img_reference, std::vector<KeyPoint> &keypoints_reference, Mat &img_target, std::vector<KeyPoint> &keypoints_target)
{

	if (siftOrSurf == 0)
	{
		Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(minHessian);
		detector->detect(img_reference, keypoints_reference);
		detector->detect(img_target, keypoints_target);
	}
	else
	{
		// SiftFeatureDetector detector = SIFT();
		Ptr<SIFT> detector = SIFT::create();
		detector->detect(img_reference, keypoints_reference);
		detector->detect(img_target, keypoints_target);
	}
}

void detect_descriptor_extractor(int siftOrSurf, Mat &img_reference, std::vector<KeyPoint> &keypoints_reference, Mat &descriptors_reference, Mat &img_target, std::vector<KeyPoint> &keypoints_target, Mat &descriptors_target)
{

	if (siftOrSurf == 0)
	{
		Ptr<xfeatures2d::SURF> extractor = xfeatures2d::SURF::create();
		extractor->compute(img_reference, keypoints_reference, descriptors_reference);
		extractor->compute(img_target, keypoints_target, descriptors_target);
	}
	else
	{
		SiftDescriptorExtractor extractor;
		extractor.compute(img_reference, keypoints_reference, descriptors_reference);
		extractor.compute(img_target, keypoints_target, descriptors_target);
	}
}

void removePontosRepetidos(std::vector<Point2f> &reference, std::vector<Point2f> &target, std::vector<Point2f> &reference_corrected, std::vector<Point2f> &target_corrected)
{
	unsigned int pt1, pt2;

	pt1 = 0;

	pt2 = 1;

	// pt3 = 0;

	int diferente = 1;

	while (pt1 < reference.size())
	{
		while ((diferente == 1) && (pt2 < reference.size()))
		{
			if ((reference[pt1].x == reference[pt2].x) && (reference[pt1].y == reference[pt2].y) && (target[pt1].x == target[pt2].x) && (target[pt1].y == target[pt2].y))
			{
				diferente = 0;
			}

			pt2 = pt2 + 1;
		}

		if (diferente == 1)
		{
			reference_corrected.push_back(Point2f((double)reference[pt1].x, (double)reference[pt1].y));
			target_corrected.push_back(Point2f((double)target[pt1].x, (double)target[pt1].y));

			// pt3 = pt3 + 1;
		}

		pt1 = pt1 + 1;

		pt2 = pt1 + 1;

		diferente = 1;
	}
}

// Seguem, abaixo, os metodos relativos ao RANSAC modificado -----------------------------------------------------

void transformaVectorEmMat(std::vector<Point2f> &vetor, Mat &matriz)
{
	for (unsigned int k = 0; k < vetor.size(); k++)
	{
		matriz.at<double>(k, 0) = vetor[k].x;
		matriz.at<double>(k, 1) = vetor[k].y;
	}
}

void moveCentroideOrigem(Mat &originalNorm, double &subtrair_eixo_x, double &subtrair_eixo_y)
{
	subtrair_eixo_x = 0.0;
	subtrair_eixo_y = 0.0;

	// no eixo x
	for (int k = 0; k < originalNorm.rows; k++)
	{
		subtrair_eixo_x = subtrair_eixo_x + originalNorm.at<double>(k, 0);
	}
	subtrair_eixo_x = subtrair_eixo_x / originalNorm.rows;
	for (int k = 0; k < originalNorm.rows; k++)
	{
		originalNorm.at<double>(k, 0) = originalNorm.at<double>(k, 0) - subtrair_eixo_x;
	}

	// no eixo y
	for (int k = 0; k < originalNorm.rows; k++)
	{
		subtrair_eixo_y = subtrair_eixo_y + originalNorm.at<double>(k, 1);
	}
	subtrair_eixo_y = subtrair_eixo_y / originalNorm.rows;
	for (int k = 0; k < originalNorm.rows; k++)
	{
		originalNorm.at<double>(k, 1) = originalNorm.at<double>(k, 1) - subtrair_eixo_y;
	}
}

double isotropicScaling(Mat &originalNorm)
{
	double dist = 0.0;

	for (int k = 0; k < originalNorm.rows; k++)
	{
		dist = dist + sqrt((originalNorm.at<double>(k, 0)) * (originalNorm.at<double>(k, 0)) + (originalNorm.at<double>(k, 1)) * (originalNorm.at<double>(k, 1)));
	}
	dist = dist / originalNorm.rows;

	if (dist == 0.0)
	{
		printf("Isotropic Scaling error! Dist == 0.0!! \n\n");
		fflush(stdout);
		//waitKey();
	}

	double scaling = sqrt(2.0) / dist;

	for (int k = 0; k < originalNorm.rows; k++)
	{
		originalNorm.at<double>(k, 0) = originalNorm.at<double>(k, 0) * scaling;
		originalNorm.at<double>(k, 1) = originalNorm.at<double>(k, 1) * scaling;
	}

	return (scaling);
}

void normalizando(Mat &original, Mat &originalNorm, Mat &T)
{

	double subtrair_eixo_x, subtrair_eixo_y;

	original.copyTo(originalNorm);

	// movendo centroide para a origem

	moveCentroideOrigem(originalNorm, subtrair_eixo_x, subtrair_eixo_y);

	// isotropic scaling

	double scaling = isotropicScaling(originalNorm);

	T.at<double>(0, 0) = scaling;
	T.at<double>(0, 1) = 0.0;
	T.at<double>(0, 2) = scaling * subtrair_eixo_x * (-1.0);
	T.at<double>(1, 0) = 0.0;
	T.at<double>(1, 1) = scaling;
	T.at<double>(1, 2) = scaling * subtrair_eixo_y * (-1.0);
	T.at<double>(2, 0) = 0.0;
	T.at<double>(2, 1) = 0.0;
	T.at<double>(2, 2) = 1.0;
}

int eInversivel(Mat &T)
{
	double isSingular;

	isSingular = determinant(T);

	double numericAdjust = 0.00000000000001;

	if ((isSingular < numericAdjust) && (isSingular > (-1.0) * numericAdjust))
	{
		printf("\n Determinante: %5.20f \n\n", isSingular);
		return (-1);
	}

	return (0);
}

void coorHomogeneas(Mat &MatNorm, Mat &MatNormHom)
{
	for (int k = 0; k < MatNorm.rows; k++)
	{
		MatNormHom.at<double>(k, 0) = MatNorm.at<double>(k, 0);
		MatNormHom.at<double>(k, 1) = MatNorm.at<double>(k, 1);
		MatNormHom.at<double>(k, 2) = 1.0;
	}
}

void generateRandomVec(Mat &PnX1, Mat &PnX2, int n, int tamanho, Mat &Ref, Mat &Tar)
{

	int controla_diferente;
	unsigned int rand_number;
	Mat rand_number_mat(n, 1, CV_8U);
	rand_number = rand() % tamanho;
	rand_number_mat.at<unsigned int>(0, 0) = rand_number;
	int varre_rand_number_mat = 1;

	for (int k = 0; k < n; k++)
	{
		PnX1.at<double>(k, 0) = Ref.at<double>(rand_number, 0);
		PnX1.at<double>(k, 1) = Ref.at<double>(rand_number, 1);
		PnX1.at<double>(k, 2) = Ref.at<double>(rand_number, 2);

		PnX2.at<double>(k, 0) = Tar.at<double>(rand_number, 0);
		PnX2.at<double>(k, 1) = Tar.at<double>(rand_number, 1);
		PnX2.at<double>(k, 2) = Tar.at<double>(rand_number, 2);

		controla_diferente = 1;

		rand_number = rand() % tamanho;

		while (controla_diferente == 1)
		{
			controla_diferente = 0;
			if (varre_rand_number_mat < n)
			{
				for (int i = 0; i < varre_rand_number_mat; i++)
				{
					if (rand_number == rand_number_mat.at<unsigned int>(i, 0))
					{
						controla_diferente = 1;
					}
				}

				if (controla_diferente == 1)
				{
					rand_number = rand() % tamanho;
				}
				else
				{
					rand_number_mat.at<int>(varre_rand_number_mat, 0) = rand_number;
					varre_rand_number_mat = varre_rand_number_mat + 1;
				}
			}
		}
	}
}

int verifyCollinearity(Mat &PnX1, int n)
{
	int isCollinear = 0;

	int degen;

	int p1, p2, p3;

	p1 = 0;
	p2 = 1;
	p3 = 2;

	while ((p1 < n - 2) && (isCollinear == 0))
	{
		while ((p2 < n - 1) && (isCollinear == 0))
		{
			// calcular equacao da reta

			double slope, intercept;

			double verificaDenominador = (PnX1.at<double>(p2, 0) - PnX1.at<double>(p1, 0));

			double var_around_zero; // variacao ao redor do zero, pois o ponto nunca estara exatamente na mesma linha - valor empirico
			var_around_zero = 0.00000000000001;

			if ((verificaDenominador < (-1.0) * var_around_zero) || (verificaDenominador > var_around_zero))
			{

				slope = (PnX1.at<double>(p2, 1) - PnX1.at<double>(p1, 1)) / (PnX1.at<double>(p2, 0) - PnX1.at<double>(p1, 0));

				intercept = PnX1.at<double>(p1, 1) - slope * PnX1.at<double>(p1, 0);

				while ((p3 < n) && (isCollinear == 0))
				{
					// testar terceiro ponto na equacao da reta

					double result;
					result = slope * PnX1.at<double>(p3, 0) + intercept - PnX1.at<double>(p3, 1);

					//		  if (result == 0)
					//			  isCollinear = 1;
					// 	Seria como esta acima se nao fosse o caso das aproximacoes
					if ((result > (-1.0) * var_around_zero) && (result < var_around_zero))
						isCollinear = 1;

					/*if (isCollinear == 1)
					{
						printf("Sem mesmo x \n\n");
						printf("Os pontos sao: \n");
						printf("x: %5.20f -- y: %5.20f \n", PnX1.at<double>(p1,0), PnX1.at<double>(p1,1));
						printf("x: %5.20f -- y: %5.20f \n", PnX1.at<double>(p2,0), PnX1.at<double>(p2,1));
						printf("x: %5.20f -- y: %5.20f \n", PnX1.at<double>(p3,0), PnX1.at<double>(p3,1));
					}*/

					p3 = p3 + 1;
				}
			}
			else
			{
				while ((p3 < n) && (isCollinear == 0))
				{
					double verificaRetaVertical; // testa se os 3 pontos tem o mesmo valor de x, sendo que seus valores de y sao indiferentes neste caso

					verificaRetaVertical = (PnX1.at<double>(p3, 0) - PnX1.at<double>(p1, 0));

					if ((verificaRetaVertical > (-1.0) * var_around_zero) && (verificaRetaVertical < var_around_zero))
					{
						isCollinear = 1;
					}

					/*if (isCollinear == 1)
					{
						printf("Com mesmo x \n\n");
						printf("Os pontos sao: \n");
						printf("x: %5.20f -- y: %5.20f \n", PnX1.at<double>(p1,0), PnX1.at<double>(p1,1));
						printf("x: %5.20f -- y: %5.20f \n", PnX1.at<double>(p2,0), PnX1.at<double>(p2,1));
						printf("x: %5.20f -- y: %5.20f \n", PnX1.at<double>(p3,0), PnX1.at<double>(p3,1));
					}*/

					p3 = p3 + 1;
				}
			}

			p2 = p2 + 1;
			p3 = p2 + 1;
		}
		p1 = p1 + 1;
		p2 = p1 + 1;
		p3 = p2 + 1;
	}

	if (isCollinear == 1)
		degen = 1;
	else
		degen = 0;

	return (degen);
}

void preencheMatrizA(Mat &A, int &ptHfit1, int &ptHfit2, Mat &PnX1Norm, Mat &PnX2Norm)
{

	A.at<double>(ptHfit1, 0) = 0.0;
	A.at<double>(ptHfit1, 1) = 0.0;
	A.at<double>(ptHfit1, 2) = 0.0;
	A.at<double>(ptHfit1, 3) = (-1.0) * PnX2Norm.at<double>(ptHfit2, 2) * PnX1Norm.at<double>(ptHfit2, 0);
	A.at<double>(ptHfit1, 4) = (-1.0) * PnX2Norm.at<double>(ptHfit2, 2) * PnX1Norm.at<double>(ptHfit2, 1);
	A.at<double>(ptHfit1, 5) = (-1.0) * PnX2Norm.at<double>(ptHfit2, 2) * PnX1Norm.at<double>(ptHfit2, 2);
	// A.at<double>(ptHfit1,6) = (-1.0)*PnX2Norm.at<double>(ptHfit2,1)*PnX1Norm.at<double>(ptHfit2,0);
	// A.at<double>(ptHfit1,7) = (-1.0)*PnX2Norm.at<double>(ptHfit2,1)*PnX1Norm.at<double>(ptHfit2,1);
	// A.at<double>(ptHfit1,8) = (-1.0)*PnX2Norm.at<double>(ptHfit2,1)*PnX1Norm.at<double>(ptHfit2,2);
	A.at<double>(ptHfit1, 6) = PnX2Norm.at<double>(ptHfit2, 1) * PnX1Norm.at<double>(ptHfit2, 0);
	A.at<double>(ptHfit1, 7) = PnX2Norm.at<double>(ptHfit2, 1) * PnX1Norm.at<double>(ptHfit2, 1);
	A.at<double>(ptHfit1, 8) = PnX2Norm.at<double>(ptHfit2, 1) * PnX1Norm.at<double>(ptHfit2, 2);

	A.at<double>(ptHfit1 + 1, 0) = PnX2Norm.at<double>(ptHfit2, 2) * PnX1Norm.at<double>(ptHfit2, 0);
	A.at<double>(ptHfit1 + 1, 1) = PnX2Norm.at<double>(ptHfit2, 2) * PnX1Norm.at<double>(ptHfit2, 1);
	A.at<double>(ptHfit1 + 1, 2) = PnX2Norm.at<double>(ptHfit2, 2) * PnX1Norm.at<double>(ptHfit2, 2);
	A.at<double>(ptHfit1 + 1, 3) = 0.0;
	A.at<double>(ptHfit1 + 1, 4) = 0.0;
	A.at<double>(ptHfit1 + 1, 5) = 0.0;
	A.at<double>(ptHfit1 + 1, 6) = (-1.0) * PnX2Norm.at<double>(ptHfit2, 0) * PnX1Norm.at<double>(ptHfit2, 0);
	A.at<double>(ptHfit1 + 1, 7) = (-1.0) * PnX2Norm.at<double>(ptHfit2, 0) * PnX1Norm.at<double>(ptHfit2, 1);
	A.at<double>(ptHfit1 + 1, 8) = (-1.0) * PnX2Norm.at<double>(ptHfit2, 0) * PnX1Norm.at<double>(ptHfit2, 2);

	A.at<double>(ptHfit1 + 2, 0) = (-1.0) * PnX2Norm.at<double>(ptHfit2, 1) * PnX1Norm.at<double>(ptHfit2, 0);
	A.at<double>(ptHfit1 + 2, 1) = (-1.0) * PnX2Norm.at<double>(ptHfit2, 1) * PnX1Norm.at<double>(ptHfit2, 1);
	A.at<double>(ptHfit1 + 2, 2) = (-1.0) * PnX2Norm.at<double>(ptHfit2, 1) * PnX1Norm.at<double>(ptHfit2, 2);
	A.at<double>(ptHfit1 + 2, 3) = PnX2Norm.at<double>(ptHfit2, 0) * PnX1Norm.at<double>(ptHfit2, 0);
	A.at<double>(ptHfit1 + 2, 4) = PnX2Norm.at<double>(ptHfit2, 0) * PnX1Norm.at<double>(ptHfit2, 1);
	A.at<double>(ptHfit1 + 2, 5) = PnX2Norm.at<double>(ptHfit2, 0) * PnX1Norm.at<double>(ptHfit2, 2);
	A.at<double>(ptHfit1 + 2, 6) = 0.0;
	A.at<double>(ptHfit1 + 2, 7) = 0.0;
	A.at<double>(ptHfit1 + 2, 8) = 0.0;
}

int HFitting(int auxHfit, Mat &PnX1, Mat &PnX2, int &degenerate, Mat &Htemp)
{
	Mat HtempAux(3, 3, DataType<double>::type);

	Mat PnX1Norm(auxHfit, 3, DataType<double>::type);
	Mat PnX2Norm(auxHfit, 3, DataType<double>::type);

	Mat T1Pn(3, 3, DataType<double>::type);
	Mat T2Pn(3, 3, DataType<double>::type);

	normalizando(PnX1, PnX1Norm, T1Pn);
	normalizando(PnX2, PnX2Norm, T2Pn);

	int invertePn = eInversivel(T2Pn);

	if (invertePn == -1)
	{
		printf("\n Matriz T2Pn não é inversível! \n\n");
		return (-1);
	}

	Mat A(3 * auxHfit, 9, DataType<double>::type);
	Mat U;
	Mat W;
	Mat vt;

	int ptHfit1 = 0;
	int ptHfit2;

	for (ptHfit2 = 0; ptHfit2 < auxHfit; ptHfit2++)
	{

		preencheMatrizA(A, ptHfit1, ptHfit2, PnX1Norm, PnX2Norm);

		ptHfit1 = ptHfit1 + 3;
	}

	SVD::compute(A, W, U, vt);

	Mat V;

	// V e a transposta de vt - V e vt sao matrizes 9x9
	if (vt.empty())
	{
		degenerate = 1;
	}
	else
	{
		V = vt.t();

		/*if (W.cols != W.rows)
		{
			printf ("A matriz W da SVD nao e quadrada!!!\n\n");
			printf("Rows: %d -- Cols: %d \n\n", W.rows, W.cols);
			return(-1);
		}*/
		double confere_autovalor = W.at<double>(0, 0);
		double confere_autovalor_aux = W.at<double>(0, 0);
		int guarda_qual_menor_autovalor = 0;
		for (int k = 0; k < W.rows; k++)
		{
			confere_autovalor_aux = W.at<double>(k, 0);
			if (confere_autovalor > confere_autovalor_aux)
			{
				confere_autovalor = confere_autovalor_aux;
				guarda_qual_menor_autovalor = k;
			}
		}
		if (guarda_qual_menor_autovalor != 8)
		{
			printf("O menor autovalor nao e o ultimo!!! \n\n");
			return (-1);
		}

		Mat h(9, 1, DataType<double>::type);

		// copiando ultima coluna de vt para h - verificar se os valores singulares de fato estao ordenados em ordem decrescente em W
		for (int k = 0; k < 9; k++)
		{
			h.at<double>(k, 0) = V.at<double>(k, 8);
		}

		Mat HauxHfit(3, 3, DataType<double>::type);

		HauxHfit.at<double>(0, 0) = h.at<double>(0, 0);
		HauxHfit.at<double>(0, 1) = h.at<double>(3, 0);
		HauxHfit.at<double>(0, 2) = h.at<double>(6, 0);
		HauxHfit.at<double>(1, 0) = h.at<double>(1, 0);
		HauxHfit.at<double>(1, 1) = h.at<double>(4, 0);
		HauxHfit.at<double>(1, 2) = h.at<double>(7, 0);
		HauxHfit.at<double>(2, 0) = h.at<double>(2, 0);
		HauxHfit.at<double>(2, 1) = h.at<double>(5, 0);
		HauxHfit.at<double>(2, 2) = h.at<double>(8, 0);

		Mat HauxHfit2(3, 3, DataType<double>::type);

		HauxHfit2 = HauxHfit.t();

		Mat T2PnInv(3, 3, DataType<double>::type);

		T2PnInv = T2Pn.inv(DECOMP_LU); // ja foi verificado anteriormente que T2Pn e inversivel

		// desnormalizando
		HtempAux = T2PnInv * HauxHfit2 * T1Pn;
	}

	int inverteHtempAux = eInversivel(HtempAux);

	if (inverteHtempAux == -1)
	{
		printf("HtempAux nao e inversivel \n\n");
		degenerate = 1;
	}
	else
	{
		HtempAux.copyTo(Htemp);
	}

	return (0);
}

void criaHX1(Mat &HX1, Mat &Htemp, Mat &referenceNormHom)
{
	for (int k = 0; k < referenceNormHom.rows; k++)
	{
		Mat referenceNormHomAux1(3, 1, DataType<double>::type);
		referenceNormHomAux1.at<double>(0, 0) = referenceNormHom.at<double>(k, 0);
		referenceNormHomAux1.at<double>(1, 0) = referenceNormHom.at<double>(k, 1);
		referenceNormHomAux1.at<double>(2, 0) = referenceNormHom.at<double>(k, 2);

		Mat referenceNormHomAux2(3, 1, DataType<double>::type);
		referenceNormHomAux2 = Htemp * referenceNormHomAux1;

		HX1.at<double>(k, 0) = referenceNormHomAux2.at<double>(0, 0);
		HX1.at<double>(k, 1) = referenceNormHomAux2.at<double>(1, 0);
		HX1.at<double>(k, 2) = referenceNormHomAux2.at<double>(2, 0);
	}
}

void criaInvHX2(Mat &invHX2, Mat &Htemp, Mat &targetNormHom)
{
	for (int k = 0; k < targetNormHom.rows; k++)
	{
		Mat targetNormHomAux1(3, 1, DataType<double>::type);
		targetNormHomAux1.at<double>(0, 0) = targetNormHom.at<double>(k, 0);
		targetNormHomAux1.at<double>(1, 0) = targetNormHom.at<double>(k, 1);
		targetNormHomAux1.at<double>(2, 0) = targetNormHom.at<double>(k, 2);

		Mat targetNormHomAux2(3, 1, DataType<double>::type);

		Mat HtempInv(3, 3, DataType<double>::type);
		HtempInv = Htemp.inv(DECOMP_LU);

		targetNormHomAux2 = HtempInv * targetNormHomAux1;

		invHX2.at<double>(k, 0) = targetNormHomAux2.at<double>(0, 0);
		invHX2.at<double>(k, 1) = targetNormHomAux2.at<double>(1, 0);
		invHX2.at<double>(k, 2) = targetNormHomAux2.at<double>(2, 0);
	}
}

void normalizaPelaTerceiraColuna(Mat &H, Mat &HNorm)
{
	for (int k = 0; k < H.rows; k++)
	{
		HNorm.at<double>(k, 0) = H.at<double>(k, 0) / H.at<double>(k, 2);
		HNorm.at<double>(k, 1) = H.at<double>(k, 1) / H.at<double>(k, 2);
		HNorm.at<double>(k, 2) = H.at<double>(k, 2) / H.at<double>(k, 2);
	}
}

int calculaD2(Mat &d2, Mat &HX1Norm, Mat &invHX2Norm, Mat &referenceNormHom, Mat &targetNormHom)
{
	if ((HX1Norm.rows != invHX2Norm.rows) || (referenceNormHom.rows != targetNormHom.rows) || (referenceNormHom.rows != HX1Norm.rows))
	{
		printf("HX1Norm, invHX2Norm, referenceNormHom e targetNormHom tem tamanhos diferentes! \n\n");
		fflush(stdout);
		//waitKey(0);
		return (-1);
	}

	double auxX1, auxY1, auxZ1, auxX2, auxY2, auxZ2;

	for (int k = 0; k < HX1Norm.rows; k++)
	{
		auxX1 = referenceNormHom.at<double>(k, 0) - invHX2Norm.at<double>(k, 0);
		auxY1 = referenceNormHom.at<double>(k, 1) - invHX2Norm.at<double>(k, 1);
		auxZ1 = referenceNormHom.at<double>(k, 2) - invHX2Norm.at<double>(k, 2);

		auxX2 = targetNormHom.at<double>(k, 0) - HX1Norm.at<double>(k, 0);
		auxY2 = targetNormHom.at<double>(k, 1) - HX1Norm.at<double>(k, 1);
		auxZ2 = targetNormHom.at<double>(k, 2) - HX1Norm.at<double>(k, 2);

		d2.at<double>(k, 0) = pow(auxX1, 2) + pow(auxY1, 2) + pow(auxZ1, 2) + pow(auxX2, 2) + pow(auxY2, 2) + pow(auxZ2, 2);

		// d2.at<double>(k,0) = (referenceNormHom.at<double>(k,0) - invHX2Norm.at<double>(k,0))*(referenceNormHom.at<double>(k,0) - invHX2Norm.at<double>(k,0)) + (referenceNormHom.at<double>(k,1) - invHX2Norm.at<double>(k,1))*(referenceNormHom.at<double>(k,1) - invHX2Norm.at<double>(k,1)) + (referenceNormHom.at<double>(k,2) - invHX2Norm.at<double>(k,2))*(referenceNormHom.at<double>(k,2) - invHX2Norm.at<double>(k,2)) + (targetNormHom.at<double>(k,0) - HX1Norm.at<double>(k,0))*(targetNormHom.at<double>(k,0) - HX1Norm.at<double>(k,0)) + (targetNormHom.at<double>(k,1) - HX1Norm.at<double>(k,1))*(targetNormHom.at<double>(k,1) - HX1Norm.at<double>(k,1)) + (targetNormHom.at<double>(k,2) - HX1Norm.at<double>(k,2))*(targetNormHom.at<double>(k,2) - HX1Norm.at<double>(k,2));

		// printf("valor de d2: %5.15f \n", d2.at<double>(k,0));
	}

	// waitKey(0);

	return (0);
}

int countingInliers(Mat &d2, double t)
{
	int counting_number_inliers = 0;
	for (int k = 0; k < d2.rows; k++)
	{

		if (d2.at<double>(k, 0) < t)
		{
			counting_number_inliers = counting_number_inliers + 1;
		}
	}

	return (counting_number_inliers);
}

void preencheInliers(Mat &inliersX1, Mat &inliersX2, Mat &inliers, Mat &referenceNormHom, Mat &targetNormHom, Mat &d2, double t)
{

	int counting_number_inliers_aux = 0;

	for (int k = 0; k < referenceNormHom.rows; k++)
	{

		if (d2.at<double>(k, 0) < t)
		{
			inliersX1.at<double>(counting_number_inliers_aux, 0) = referenceNormHom.at<double>(k, 0);
			inliersX1.at<double>(counting_number_inliers_aux, 1) = referenceNormHom.at<double>(k, 1);
			inliersX1.at<double>(counting_number_inliers_aux, 2) = referenceNormHom.at<double>(k, 2);

			inliersX2.at<double>(counting_number_inliers_aux, 0) = targetNormHom.at<double>(k, 0);
			inliersX2.at<double>(counting_number_inliers_aux, 1) = targetNormHom.at<double>(k, 1);
			inliersX2.at<double>(counting_number_inliers_aux, 2) = targetNormHom.at<double>(k, 2);

			inliers.at<double>(counting_number_inliers_aux, 0) = referenceNormHom.at<double>(k, 0);
			inliers.at<double>(counting_number_inliers_aux, 1) = referenceNormHom.at<double>(k, 1);
			inliers.at<double>(counting_number_inliers_aux, 2) = referenceNormHom.at<double>(k, 2);
			inliers.at<double>(counting_number_inliers_aux, 3) = targetNormHom.at<double>(k, 0);
			inliers.at<double>(counting_number_inliers_aux, 4) = targetNormHom.at<double>(k, 1);
			inliers.at<double>(counting_number_inliers_aux, 5) = targetNormHom.at<double>(k, 2);

			counting_number_inliers_aux = counting_number_inliers_aux + 1;
		}
	}

	if (counting_number_inliers_aux != inliersX1.rows)
	{
		printf("Erro no tamanho de inliersX1! \n\n");
		fflush(stdout);
		// waitKey(0);
	}
}

void calculaMatrizCovariancia(Mat &inliers, Mat &covar)
{
	double mediaIn_x = 0.0;
	double mediaIn_y = 0.0;
	double mediaIn_z = 0.0;

	for (int k = 0; k < inliers.rows; k++)
	{
		mediaIn_x = mediaIn_x + inliers.at<double>(k, 0);
		mediaIn_y = mediaIn_y + inliers.at<double>(k, 1);
		mediaIn_z = mediaIn_z + inliers.at<double>(k, 2);
	}

	mediaIn_x = mediaIn_x / inliers.rows;
	mediaIn_y = mediaIn_y / inliers.rows;
	mediaIn_z = mediaIn_z / inliers.rows;

	Mat inliersAuxCov;

	inliers.copyTo(inliersAuxCov);

	for (int k = 0; k < inliersAuxCov.rows; k++)
	{
		inliersAuxCov.at<double>(k, 0) = inliersAuxCov.at<double>(k, 0) - mediaIn_x;
		inliersAuxCov.at<double>(k, 1) = inliersAuxCov.at<double>(k, 1) - mediaIn_y;
		inliersAuxCov.at<double>(k, 2) = inliersAuxCov.at<double>(k, 2) - mediaIn_z;
	}
	double counting_number_inliers_aux_double = (double)inliers.rows;
	covar = (1.0 / (counting_number_inliers_aux_double - 1.0)) * inliersAuxCov.t() * inliersAuxCov;
}

void calculaMatrizMaiorCovariancia(Mat &inliers, Mat &covar)
{
	double mediaIn_x = 0.0;
	double mediaIn_y = 0.0;
	double mediaIn_z = 0.0;
	double mediaIn_k = 0.0;
	double mediaIn_w = 0.0;
	double mediaIn_f = 0.0;

	for (int k = 0; k < inliers.rows; k++)
	{
		mediaIn_x = mediaIn_x + inliers.at<double>(k, 0);
		mediaIn_y = mediaIn_y + inliers.at<double>(k, 1);
		mediaIn_z = mediaIn_z + inliers.at<double>(k, 2);
		mediaIn_k = mediaIn_k + inliers.at<double>(k, 3);
		mediaIn_w = mediaIn_w + inliers.at<double>(k, 4);
		mediaIn_f = mediaIn_f + inliers.at<double>(k, 5);
	}

	mediaIn_x = mediaIn_x / inliers.rows;
	mediaIn_y = mediaIn_y / inliers.rows;
	mediaIn_z = mediaIn_z / inliers.rows;
	mediaIn_k = mediaIn_k / inliers.rows;
	mediaIn_w = mediaIn_w / inliers.rows;
	mediaIn_f = mediaIn_f / inliers.rows;

	Mat inliersAuxCov;

	inliers.copyTo(inliersAuxCov);

	for (int k = 0; k < inliersAuxCov.rows; k++)
	{
		inliersAuxCov.at<double>(k, 0) = inliersAuxCov.at<double>(k, 0) - mediaIn_x;
		inliersAuxCov.at<double>(k, 1) = inliersAuxCov.at<double>(k, 1) - mediaIn_y;
		inliersAuxCov.at<double>(k, 2) = inliersAuxCov.at<double>(k, 2) - mediaIn_z;
		inliersAuxCov.at<double>(k, 3) = inliersAuxCov.at<double>(k, 3) - mediaIn_k;
		inliersAuxCov.at<double>(k, 4) = inliersAuxCov.at<double>(k, 4) - mediaIn_w;
		inliersAuxCov.at<double>(k, 5) = inliersAuxCov.at<double>(k, 5) - mediaIn_f;
	}
	double counting_number_inliers_aux_double = (double)inliers.rows;
	covar = (1.0 / (counting_number_inliers_aux_double - 1.0)) * inliersAuxCov.t() * inliersAuxCov;
}

double calculaScatter(std::vector<double> &eigenvalues)
{
	double scatterInliers = eigenvalues[0];
	for (unsigned int k = 0; k < eigenvalues.size(); k++)
	{
		if (scatterInliers < eigenvalues[k])
			scatterInliers = eigenvalues[k];
	}

	return (scatterInliers);
}

void geraImagemDupla(Mat &img_reference, Mat &img_target, Mat &imgDuplaColor)
{

	Mat imgDupla(img_reference.rows, img_reference.cols + img_target.cols, DataType<unsigned char>::type);

	for (int k = 0; k < img_reference.rows; k++)
	{
		for (int l = 0; l < img_reference.cols; l++)
		{
			imgDupla.at<unsigned char>(k, l) = img_reference.at<unsigned char>(k, l);
		}
	}

	for (int k = 0; k < img_reference.rows; k++)
	{
		for (int l = img_reference.cols; l < (img_reference.cols + img_target.cols); l++)
		{
			imgDupla.at<unsigned char>(k, l) = img_target.at<unsigned char>(k, l - img_target.cols);
		}
	}

	cvtColor(imgDupla, imgDuplaColor, COLOR_GRAY2BGR);
}

void desnormaliza(Mat &Tinv, Mat &bestInliersX, std::vector<Point2f> &bestInXVec)
{
	for (int k = 0; k < bestInliersX.rows; k++)
	{
		Mat aux1(3, 1, DataType<double>::type);
		Mat aux2(3, 1, DataType<double>::type);

		aux1.at<double>(0, 0) = bestInliersX.at<double>(k, 0);
		aux1.at<double>(1, 0) = bestInliersX.at<double>(k, 1);
		aux1.at<double>(2, 0) = bestInliersX.at<double>(k, 2);

		aux2 = Tinv * aux1;

		Point2f pt;

		pt.x = aux2.at<double>(0, 0);
		pt.y = aux2.at<double>(1, 0);

		bestInXVec.push_back(pt);
	}
}

void desenhando(Mat &imgDupla, std::vector<Point2f> &bestInX1Vec, std::vector<Point2f> &bestInX2Vec)
{
	for (unsigned int k = 0; k < bestInX1Vec.size(); k++)
	{
		Point2f ponto1, ponto2;

		unsigned int rand_number1, rand_number2, rand_number3;
		rand_number1 = rand() % 255;
		rand_number2 = rand() % 255;
		rand_number3 = rand() % 255;

		Scalar corLinha = CV_RGB(rand_number1, rand_number2, rand_number3);

		ponto1.x = bestInX1Vec[k].x;
		ponto1.y = bestInX1Vec[k].y;

		ponto2.x = bestInX2Vec[k].x + +((imgDupla.cols) / 2.0);
		ponto2.y = bestInX2Vec[k].y;

		line(imgDupla, ponto1, ponto2, corLinha, 2);
	}
}

void desenhaPontos(Mat &imgDupla, Mat &T1, Mat &T2, Mat &bestInliersX1, Mat &bestInliersX2)
{
	std::vector<Point2f> bestInX1Vec;
	std::vector<Point2f> bestInX2Vec;

	Mat T1inv(3, 3, DataType<double>::type);
	Mat T2inv(3, 3, DataType<double>::type);

	T1inv = T1.inv(DECOMP_LU);
	T2inv = T2.inv(DECOMP_LU);

	desnormaliza(T1inv, bestInliersX1, bestInX1Vec);

	desnormaliza(T2inv, bestInliersX2, bestInX2Vec);

	desenhando(imgDupla, bestInX1Vec, bestInX2Vec);
}

int modifiedHomography(Mat &Hinter, std::vector<Point2f> &reference, std::vector<Point2f> &target, Mat &imgDupla)
{
	Mat referenceMat(reference.size(), 2, DataType<double>::type);
	Mat targetMat(target.size(), 2, DataType<double>::type);

	if (reference.size() != target.size())
	{
		printf("Tamanho dos vetores reference e target e diferente! \n\n");
		fflush(stdout);
		return (-1);
	}

	transformaVectorEmMat(reference, referenceMat);
	transformaVectorEmMat(target, targetMat);

	// printf("Reference: tamanho vetor: %lu -- tamanho matriz: %d \n\n", reference.size(), referenceMat.rows);
	// printf("Target: tamanho vetor: %lu -- tamanho matriz: %d \n\n", target.size(), targetMat.rows);
	// imprimeMatriz(referenceMat);
	// imprimeMatriz(targetMat);
	// waitKey(0);

	Mat referenceMatNorm(referenceMat.rows, 2, DataType<double>::type);
	Mat targetMatNorm(targetMat.rows, 2, DataType<double>::type);

	Mat T1(3, 3, DataType<double>::type);
	Mat T2(3, 3, DataType<double>::type);

	// Normalizando pontos --------------------------------------------------------------------------------------

	normalizando(referenceMat, referenceMatNorm, T1);
	normalizando(targetMat, targetMatNorm, T2);

	int inverte = eInversivel(T2);

	if (inverte == -1)
	{
		printf("\n Matriz T2 não é inversível! \n\n");
		return (-1);
	}

	// coordenadas homogeneas ------------------------------------------------------------------------------------

	Mat referenceNormHom(referenceMatNorm.rows, 3, DataType<double>::type);
	Mat targetNormHom(targetMatNorm.rows, 3, DataType<double>::type);

	coorHomogeneas(referenceMatNorm, referenceNormHom);
	coorHomogeneas(targetMatNorm, targetNormHom);

	// printf("Reference: tamanho matriz: %d \n\n", referenceNormHom.rows);
	// printf("Target: tamanho matriz: %d \n\n", targetNormHom.rows);
	// waitKey(0);

	// Variaveis usadas somente para Debug ...
	// int contador_loop_externo = 0;
	// int contador_loop_interno = 0;

	// ------------------------------------------- mRansac -------------------------------------------------------------

	// Zisserman pg 119
	double p = 0.99;

	// n -> minimal number of data required to fit the model
	int n = 4;

	// maximum number of iterations allowed
	int mI = 500;

	// maximum number of trials to select a non-degenerate data set
	int mInd = 100;

	// threshold for determining when a datum fits a model
	double t = 0.001;
	// double t = 0.0005;
	// double t = 0.000001;
	// double t = 10.0;

	int trialcount = 0;

	// int trialcountND = 0;			// nao e usado para nada...

	Mat bestH;

	// int N = 1;
	// double N = 1;
	double N = 10000000000.0;

	Mat inliers;

	Mat inliersX1;
	Mat inliersX2;

	double maxScore = 0.0;

	double ninliers;

	Mat bestInliers;

	Mat bestInliersX1;
	Mat bestInliersX2;

	int degenerate;

	int count;

	srand(time(NULL)); // Deve ficar aqui, ou no programa principal ??????????????????????????????????????????????????? --- aqui...

	if (referenceNormHom.rows < 4)
	{
		printf("Menos de 4 pontos - nao e possivel fazer homografia \n\n");
		return (-1);
	}

	while ((N > trialcount) && (trialcount <= mI))
	// while (trialcount <= mI)
	{

		// Debug, apenas...
		// contador_loop_externo++;
		// printf("Contador loop externo: %d \n\n", contador_loop_externo);
		// fflush(stdout);

		degenerate = 1; // is degenerate
		count = 1;

		Mat Htemp(3, 3, DataType<double>::type);

		while ((degenerate == 1) && (count <= mInd))
		{
			// Debug, apenas...
			// contador_loop_interno++;
			// printf("Contador loop interno: %d \n\n", contador_loop_interno);
			// fflush(stdout);

			// Randomly sample n pairs of normalized data

			Mat PnX1(n, 3, DataType<double>::type);
			Mat PnX2(n, 3, DataType<double>::type);

			generateRandomVec(PnX1, PnX2, n, referenceNormHom.rows, referenceNormHom, targetNormHom);

			// verify collinearity of the n points

			// so se deve testar colinearidade dos 4 pontos da imagem de referencia!!!!!!!

			degenerate = verifyCollinearity(PnX1, n);

			// if there are not 3 points of this set in a line, degenerate = 0
			if (degenerate == 0)
			{
				// call H-Fitting -------------------------------------------

				int auxHfit = n; // n, neste caso; e tamanho do vetor bestInliers, para obter a matriz final resultante do mRansac

				int confereHfitting = HFitting(auxHfit, PnX1, PnX2, degenerate, Htemp);

				if (confereHfitting == -1)
				{
					printf("Problemas com o Hfitting!! No loop interno!/n/n");
					return (-1);
				}
			}
			count = count + 1;
		}
		if (degenerate == 1)
		{
			trialcount = trialcount + 1;
		}
		else
		{
			// o resto do algoritmo mRansac aqui - exceto do ultimo H-Fitting em diante
			// evaluate Htemp by the inliers matching Htemp

			Mat HX1(referenceNormHom.rows, 3, DataType<double>::type);
			criaHX1(HX1, Htemp, referenceNormHom);

			Mat invHX2(targetNormHom.rows, 3, DataType<double>::type);
			criaInvHX2(invHX2, Htemp, targetNormHom);

			Mat HX1Norm(referenceNormHom.rows, 3, DataType<double>::type);
			normalizaPelaTerceiraColuna(HX1, HX1Norm);

			Mat invHX2Norm(targetNormHom.rows, 3, DataType<double>::type);
			normalizaPelaTerceiraColuna(invHX2, invHX2Norm);

			Mat d2(HX1Norm.rows, 1, DataType<double>::type);

			// printf("Tamanhos: d2: %d -- HX1Norm: %d -- invHX2Norm: %d \n\n", d2.rows, HX1Norm.rows, invHX2Norm.rows);
			// waitKey();

			int verificaD2 = calculaD2(d2, HX1Norm, invHX2Norm, referenceNormHom, targetNormHom);

			if (verificaD2 == -1)
			{
				printf("Erro no calculo de d2!\n\n");
				fflush(stdout);
				return (-1);
			}

			int counting_number_inliers = countingInliers(d2, t);

			inliers.create(counting_number_inliers, 6, DataType<double>::type);
			inliersX1.create(counting_number_inliers, 3, DataType<double>::type);
			inliersX2.create(counting_number_inliers, 3, DataType<double>::type);

			if (counting_number_inliers > 3)
			{
				preencheInliers(inliersX1, inliersX2, inliers, referenceNormHom, targetNormHom, d2, t);

				// printf("Imprimindo vetor InliersX1\n\n");
				// imprimeMatriz(inliersX1);
				// waitKey(0);

				Mat covar;
				Mat covarX1;
				Mat covarX2;

				calculaMatrizCovariancia(inliersX1, covarX1);

				// parte abaixo apenas para teste, mesmo... ---------------------------------------
				if (covarX1.rows != 3)
				{
					printf("Linhas em inliersX1: %d \n\n", inliersX1.rows);
					printf("Dimensao covarX1: %d", covarX1.rows);
					//waitKey(0);
					return (-1);
				}
				// ---------------------------------------------------------------------------------

				calculaMatrizCovariancia(inliersX2, covarX2);

				// usada so para comparacao entre os valores dos espalhamentos - seu tamanho e 6x6, dai o MAIOR no nome da funcao
				calculaMatrizMaiorCovariancia(inliers, covar);

				// printf("Imprimindo matriz CovarX1: \n\n");
				// waitKey(0);

				// imprimeMatriz(covarX1);

				std::vector<double> eigenvalues;
				std::vector<double> eigenvaluesX1;
				std::vector<double> eigenvaluesX2;

				eigen(covarX1, eigenvaluesX1);
				eigen(covarX2, eigenvaluesX2);
				eigen(covar, eigenvalues);

				double scatterInliers = calculaScatter(eigenvalues);
				double scatterInliersX1 = calculaScatter(eigenvaluesX1);
				double scatterInliersX2 = calculaScatter(eigenvaluesX2);

				// printf ("\n\n ScatterInliers: %lf, ScatterInliersX1: %lf, ScatterInliersx2: %lf \n\n", scatterInliers, scatterInliersX1, scatterInliersX2);
				// printf("\n\n Esses valores teriam que ser parecidos... \n\n");
				fflush(stdout);
				// waitKey(0);

				double counting_number_inliers_double;

				counting_number_inliers_double = (double)counting_number_inliers;

				ninliers = 0.0;

				// ninliers = counting_number_inliers_double*scatterInliersX1;
				ninliers = counting_number_inliers_double;
				// ninliers = counting_number_inliers_double*scatterInliersX2;
				// ninliers = counting_number_inliers_double*scatterInliers;

				if (ninliers > maxScore)
				{
					maxScore = ninliers;
					inliers.copyTo(bestInliers);
					inliersX1.copyTo(bestInliersX1);
					inliersX2.copyTo(bestInliersX2);

					Htemp.copyTo(bestH);

					double fracinliers = ninliers / referenceNormHom.rows;
					double pNoOutliers = 1.0 - pow(fracinliers, n);
					N = log10(1.0 - p) / log10(pNoOutliers);
				}
			}

			trialcount = trialcount + 1;
		}
	}

	// verificar aqui se bestH e ou nao empty... Se for, nao fazer mais nada no resto do codigo inteiro, e dar mensagem de erro!
	if (bestH.empty())
	{
		printf("Nao foi produzida nenhuma matriz de homografia no mRansac - bestH esta vazio!! \n\n");
		return (-1);
	}

	// printf("Imprimindo matriz bestH: \n\n");
	// imprimeMatriz(bestH);
	fflush(stdout);

	// call H-Fitting based on bestInliers - returns Mat Htemp2
	Mat Htemp2(3, 3, DataType<double>::type);

	// printf("Imprimindo vetor bestInliersX1\n\n");
	// imprimeMatriz(bestInliersX1);
	fflush(stdout);

	if (bestInliersX1.rows < 4)
	{
		printf("Menos de 4 pontos: impossivel calcular homografia! \n\n");
		return (-1);
	}

	// printf("\n");
	fflush(stdout);

	Mat PnX1NormFinal;
	Mat PnX2NormFinal;

	bestInliersX1.copyTo(PnX1NormFinal);
	bestInliersX2.copyTo(PnX2NormFinal);

	int degenFinal = 0;

	int auxHfitFinal = bestInliersX1.rows; // n, neste caso; e tamanho do vetor bestInliers, para obter a matriz final resultante do mRansac

	int confereHfittingFinal = HFitting(auxHfitFinal, PnX1NormFinal, PnX2NormFinal, degenFinal, Htemp2);

	if (confereHfittingFinal == -1)
	{
		printf("Problemas com o HfittingFinal!!/n/n");
		return (-1);
	}

	if (degenFinal == 1)
	{
		printf("Problemas com o HfittingFinal!! Matriz vt vazia!! /n/n");
		return (-1);
	}

	Mat invT2 = T2.inv(DECOMP_LU);

	Hinter = invT2 * Htemp2 * T1;

	desenhaPontos(imgDupla, T1, T2, bestInliersX1, bestInliersX2);

	return (0);
}

int calculaNCCImage(Mat &warpImg, Mat &img_target, Mat &NCC_img_8b, int dimensao, Mat &warpImgMask)
{
	// Criando NCC Image

	Mat NCC_img;

	// NCC_img.create(img_object.size(), img_object.type());
	NCC_img.create(img_target.size(), DataType<double>::type);

	int contador_imagem_linha;
	int contador_imagem_coluna;

	int varre_imgAux_linha;
	int varre_imgAux_coluna;

	// variaveis auxiliares para varrer imagens e preencher matrizes auxiliares de dimensao NxN
	int varre_imagem_linha;
	int varre_imagem_coluna;

	// dimensao deve ser impar, para haver um elemento central na matriz
	// int dimensao = atoi (argv[3]);

	// verificando se dimensao e impar
	if (dimensao % 2 == 0)
		return (2);

	// verificando se dimensao e no minimo 3x3
	if (dimensao <= 1)
		return (2);

	int deslocamento = (dimensao - 1) / 2;

	printf("\n Deslocamento: %d \n\n", deslocamento);

	// Mat auxMat_R (dimensao, dimensao, img_object.type());
	// Mat auxMat_T (dimensao, dimensao, img_object.type());

	Mat auxMat_R(dimensao, dimensao, DataType<double>::type);
	Mat auxMat_T(dimensao, dimensao, DataType<double>::type);

	// printf ("\n Depois de criar matrizes auxiliares \n\n");

	// int auxcol = warpImg.cols;
	// int auxlin = warpImg.rows;

	// printf ("\n Linhas: %d , Colunas: %d \n\n", auxlin, auxcol);

	for (contador_imagem_linha = 0; contador_imagem_linha < warpImg.rows; contador_imagem_linha++)
	{
		for (contador_imagem_coluna = 0; contador_imagem_coluna < warpImg.cols; contador_imagem_coluna++)
		{

			varre_imagem_linha = contador_imagem_linha - deslocamento;
			varre_imagem_coluna = contador_imagem_coluna - deslocamento;

			for (varre_imgAux_linha = 0; varre_imgAux_linha < dimensao; varre_imgAux_linha++)
			{
				for (varre_imgAux_coluna = 0; varre_imgAux_coluna < dimensao; varre_imgAux_coluna++)
				{

					if (varre_imagem_linha < 0 || varre_imagem_coluna < 0 || varre_imagem_linha >= warpImg.rows || varre_imagem_coluna >= warpImg.cols)
					{
						auxMat_R.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = 0;
						auxMat_T.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = 0;
					}
					else
					{
						auxMat_R.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = warpImg.at<unsigned char>(varre_imagem_linha, varre_imagem_coluna);
						auxMat_T.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = img_target.at<unsigned char>(varre_imagem_linha, varre_imagem_coluna);
					}
					varre_imagem_coluna++;
				}

				varre_imgAux_coluna = 0;

				varre_imagem_coluna = contador_imagem_coluna - deslocamento;
				varre_imagem_linha++;
			}

			// Duas matrizes auxiliares NxN formadas
			// Calcular correlacao cruzada normalizada

			double media_Raux = 0;
			double media_Taux = 0;

			double norma_Raux = 0;
			double norma_Taux = 0;

			for (int cont_matAux_lin = 0; cont_matAux_lin < auxMat_R.rows; cont_matAux_lin++)
			{
				for (int cont_matAux_col = 0; cont_matAux_col < auxMat_R.cols; cont_matAux_col++)
				{
					media_Raux = media_Raux + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col);
					media_Taux = media_Taux + auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);

					// norma_Raux = norma_Raux + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col)*auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col);
					// norma_Taux = norma_Taux + auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col)*auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);
				}
			}
			media_Raux = media_Raux / (auxMat_R.rows * auxMat_R.cols);
			media_Taux = media_Taux / (auxMat_T.rows * auxMat_T.cols);

			// norma_Raux = sqrt(norma_Raux);
			// norma_Taux = sqrt(norma_Taux);

			double pixel_NCC = 0;

			for (int cont_matAux_lin = 0; cont_matAux_lin < auxMat_R.rows; cont_matAux_lin++)
			{
				for (int cont_matAux_col = 0; cont_matAux_col < auxMat_R.cols; cont_matAux_col++)
				{
					auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) = (auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) - media_Raux);
					auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) = (auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) - media_Taux);

					// pixel_NCC = pixel_NCC + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col)*auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);
				}
			}

			for (int cont_matAux_lin = 0; cont_matAux_lin < auxMat_R.rows; cont_matAux_lin++)
			{
				for (int cont_matAux_col = 0; cont_matAux_col < auxMat_R.cols; cont_matAux_col++)
				{
					// media_Raux = media_Raux + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col);
					// media_Taux = media_Taux + auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);

					norma_Raux = norma_Raux + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) * auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col);
					norma_Taux = norma_Taux + auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) * auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);
				}
			}

			norma_Raux = sqrt(norma_Raux);
			norma_Taux = sqrt(norma_Taux);

			for (int cont_matAux_lin = 0; cont_matAux_lin < auxMat_R.rows; cont_matAux_lin++)
			{
				for (int cont_matAux_col = 0; cont_matAux_col < auxMat_R.cols; cont_matAux_col++)
				{
					auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) = auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) / norma_Raux;
					auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) = auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) / norma_Taux;

					pixel_NCC = pixel_NCC + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) * auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);
				}
			}

			// if (pixel_NCC < 0)
			//   pixel_NCC = 0;

			// if (pixel_NCC > 255)
			//  pixel_NCC = 255;

			// preencher elemento correspondente na NCC_img

			NCC_img.at<double>(contador_imagem_linha, contador_imagem_coluna) = pixel_NCC;
		}

		contador_imagem_coluna = 0;
	}

	// printf("\n Antes de exibir imagem NCC \n\n");

	for (int k = 0; k < NCC_img.rows; k++)
	{
		for (int l = 0; l < NCC_img.cols; l++)
		{
			if ((double)warpImgMask.at<unsigned char>(k, l) == 0.0)
			{
				NCC_img.at<double>(k, l) = -127.5;
			}
		}
	}

	// Mat NCC_img_8b;

	// NCC_img_8b.create(img_object.size(), img_object.type());

	// cvConvertScale (&NCC_img, &NCC_img_8b);
	NCC_img.convertTo(NCC_img_8b, NCC_img_8b.type(), 127.5, 127.5);

	// exibir NCC_img

	// matchTemplate(img_scene, warpImg, NCC_img, CV_TM_CCORR_NORMED);

	// imshow("NCC", NCC_img_8b);

	return (0);
}

void geraInicioFilaHistogramaAngulo(std::vector<Point2f> &reference_corrected, std::vector<Point2f> &target_corrected, structAngulo &inicial, double &dist)
{
	unsigned int tamanho = reference_corrected.size();
	Point2f elementoAux;

	// calcula angulos
	for (unsigned int k = 0; k < tamanho; k++)
	{
		elementoAux.y = 0.0;
		if (target_corrected[k].x == reference_corrected[k].x)
		{
			elementoAux.x = 9999999999.9999999999;
		}
		else
		{
			elementoAux.x = (target_corrected[k].y - reference_corrected[k].y) / (dist + target_corrected[k].x - reference_corrected[k].x);
		}
		inicial.angulo.push_back(elementoAux);
	}

	// tamanho do vetor
	inicial.tamanhoVetor = tamanho;

	// calcula media dos angulos
	double mediaAux = 0.0;

	for (unsigned int k = 0; k < tamanho; k++)
	{
		mediaAux = mediaAux + inicial.angulo[k].x;
	}

	mediaAux = mediaAux / (double)tamanho;

	inicial.media = mediaAux;

	// calcula desvio padrao
	for (unsigned int k = 0; k < tamanho; k++)
	{
		inicial.angulo[k].y = inicial.angulo[k].x - mediaAux;
	}

	double desvioPadraoAux = 0.0;

	if (tamanho < 2)
	{
		inicial.desvioPadrao = 0.0;
	}
	else
	{
		for (unsigned int k = 0; k < tamanho; k++)
		{
			desvioPadraoAux = desvioPadraoAux + (inicial.angulo[k].y * inicial.angulo[k].y);
		}

		desvioPadraoAux = desvioPadraoAux * (1.0 / (tamanho - 1.0));

		desvioPadraoAux = sqrt(desvioPadraoAux);

		inicial.desvioPadrao = desvioPadraoAux;
	}

	inicial.anterior = NULL;
	inicial.posterior = NULL;
}

void imprimeElementoHistogramaAngulo(structAngulo &inicial)
{
	printf("\n");
	printf("Coef angular medio: %5.20f -- ", inicial.media);
	printf("Desvio padrao: %5.20f -- ", inicial.desvioPadrao);
	printf("Tamanho do vetor: %d ", inicial.tamanhoVetor);
	printf("\n\n");
	fflush(stdout);
}

void geraFilaHistogramaAngulo(int quantidade, structAngulo &inicial, structAngulo *inicioFila)
{
	int contador = 1;

	int tamanhoVetorAux = 0;

	structAngulo *structMaiorTamanhoAux, *varreFila, *anteriorAux, *posteriorAux;

	varreFila = inicioFila;

	structMaiorTamanhoAux = inicioFila;

	anteriorAux = NULL;
	posteriorAux = NULL;

	while (contador < quantidade)
	{
		while (varreFila != NULL)
		{
			// coloca ponteiro auxiliar apontando para estrutura com maior numero de elementos no vetor
			if (varreFila->tamanhoVetor > tamanhoVetorAux)
			{
				tamanhoVetorAux = varreFila->tamanhoVetor;

				structMaiorTamanhoAux = varreFila;
			}
			varreFila = varreFila->posterior;
		}
		anteriorAux = structMaiorTamanhoAux->anterior;
		posteriorAux = structMaiorTamanhoAux->posterior;

		structAngulo auxiliarA, auxiliarB;

		auxiliarA.anterior = anteriorAux;
		auxiliarA.posterior = &auxiliarB;
		auxiliarB.anterior = &auxiliarA;
		auxiliarB.posterior = posteriorAux;

		if (anteriorAux != NULL)
		{
			anteriorAux->posterior = &auxiliarA;
		}
		if (posteriorAux != NULL)
		{
			posteriorAux->anterior = &auxiliarB;
		}
		if (inicioFila == structMaiorTamanhoAux)
		{
			inicioFila = &auxiliarA;
		}

		Point2f aux1;

		for (int k = 0; k < structMaiorTamanhoAux->tamanhoVetor; k++)
		{
			aux1.y = 0.0;
			aux1.x = structMaiorTamanhoAux->angulo[k].x;

			if (aux1.x < structMaiorTamanhoAux->media)
			{
				auxiliarA.angulo.push_back(aux1);
			}
			else
			{
				auxiliarB.angulo.push_back(aux1);
			}
		}

		auxiliarA.tamanhoVetor = auxiliarA.angulo.size();
		auxiliarB.tamanhoVetor = auxiliarB.angulo.size();

		// calcula media dos angulos
		double mediaAuxA = 0.0;

		for (int k = 0; k < auxiliarA.tamanhoVetor; k++)
		{
			mediaAuxA = mediaAuxA + auxiliarA.angulo[k].x;
		}

		mediaAuxA = mediaAuxA / (double)auxiliarA.tamanhoVetor;

		auxiliarA.media = mediaAuxA;

		// calcula media dos angulos
		double mediaAuxB = 0.0;

		for (int k = 0; k < auxiliarB.tamanhoVetor; k++)
		{
			mediaAuxB = mediaAuxB + auxiliarB.angulo[k].x;
		}

		mediaAuxB = mediaAuxB / (double)auxiliarB.tamanhoVetor;

		auxiliarB.media = mediaAuxB;

		// calcula desvio padrao
		for (int k = 0; k < auxiliarA.tamanhoVetor; k++)
		{
			auxiliarA.angulo[k].y = auxiliarA.angulo[k].x - auxiliarA.media;
		}

		double desvioPadraoAuxA = 0.0;

		if (auxiliarA.tamanhoVetor < 2)
		{
			auxiliarA.desvioPadrao = 0.0;
		}
		else
		{
			for (int k = 0; k < auxiliarA.tamanhoVetor; k++)
			{
				desvioPadraoAuxA = desvioPadraoAuxA + (auxiliarA.angulo[k].y * auxiliarA.angulo[k].y);
			}

			desvioPadraoAuxA = desvioPadraoAuxA * (1.0 / (auxiliarA.tamanhoVetor - 1.0));

			desvioPadraoAuxA = sqrt(desvioPadraoAuxA);

			auxiliarA.desvioPadrao = desvioPadraoAuxA;
		}

		// calcula desvio padrao
		for (int k = 0; k < auxiliarB.tamanhoVetor; k++)
		{
			auxiliarB.angulo[k].y = auxiliarB.angulo[k].x - auxiliarB.media;
		}

		double desvioPadraoAuxB = 0.0;

		if (auxiliarB.tamanhoVetor < 2)
		{
			auxiliarB.desvioPadrao = 0.0;
		}
		else
		{
			for (int k = 0; k < auxiliarB.tamanhoVetor; k++)
			{
				desvioPadraoAuxB = desvioPadraoAuxB + (auxiliarB.angulo[k].y * auxiliarB.angulo[k].y);
			}

			desvioPadraoAuxB = desvioPadraoAuxB * (1.0 / (auxiliarB.tamanhoVetor - 1.0));

			desvioPadraoAuxB = sqrt(desvioPadraoAuxB);

			auxiliarB.desvioPadrao = desvioPadraoAuxB;
		}

		contador++;
	}
}

void removeAnguloGrande(std::vector<Point2f> &reference_corrected, std::vector<Point2f> &target_corrected, std::vector<Point2f> &reference_angulo, std::vector<Point2f> &target_angulo, double &controlaIntervaloAngulo, double &dist)
{
	unsigned int tamanho = reference_corrected.size();
	double angulo;
	Point2f elementoAux1, elementoAux2;

	// calcula angulos
	for (unsigned int k = 0; k < tamanho; k++)
	{
		if (target_corrected[k].x == reference_corrected[k].x)
		{
			// deveria ser infinito...
			angulo = 999999999.9999999999;
		}
		else
		{
			angulo = (target_corrected[k].y - reference_corrected[k].y) / (dist + target_corrected[k].x - reference_corrected[k].x);

			// printf("Coef angular: %5.20f \n", angulo);
			// fflush(stdout);
		}

		if ((angulo > (-1.0) * controlaIntervaloAngulo) && (angulo < controlaIntervaloAngulo))
		{
			elementoAux1.x = reference_corrected[k].x;
			elementoAux1.y = reference_corrected[k].y;
			reference_angulo.push_back(elementoAux1);

			elementoAux2.x = target_corrected[k].x;
			elementoAux2.y = target_corrected[k].y;
			target_angulo.push_back(elementoAux2);
		}
	}

	/*
		printf("\n");
		fflush(stdout);
		for (unsigned int k = 0; k < reference_angulo.size(); k++)
		{
			printf("Xref: %5.20f, Yref: %5.20f -- Xtar: %5.20f, Ytar: %5.20f \n", reference_angulo[k].x, reference_angulo[k].y, target_angulo[k].x, target_angulo[k].y);
			fflush(stdout);
		}
		printf("\n");
		fflush(stdout);
	*/
}

void removeDistanciaDispar(std::vector<Point2f> &reference_angulo, std::vector<Point2f> &target_angulo, std::vector<Point2f> &reference_distancia, std::vector<Point2f> &target_distancia, double &dist, double &intervaloTamanho, int &soEixoX)
{

	std::vector<double> comprimento;
	double desvPadAux = 0.0;

	for (unsigned int k = 0; k < reference_angulo.size(); k++)
	{
		double comprimentoAux;
		double b2, c2;

		if (soEixoX == 1)
		{
			comprimentoAux = (reference_angulo[k].x - target_angulo[k].x + dist);
		}
		else
		{
			b2 = (reference_angulo[k].y - target_angulo[k].y);
			b2 = b2 * b2;

			c2 = (reference_angulo[k].x - target_angulo[k].x + dist);
			c2 = c2 * c2;

			comprimentoAux = sqrt(b2 + c2);
		}

		comprimento.push_back(comprimentoAux);
	}

	double media = 0.0;

	for (unsigned int k = 0; k < comprimento.size(); k++)
	{
		media = media + comprimento[k];
	}
	media = media / comprimento.size();

	if (comprimento.size() > 1)
	{
		for (unsigned int k = 0; k < comprimento.size(); k++)
		{
			double desv = 0.0;
			desv = comprimento[k] - media;
			desv = desv * desv;
			desvPadAux = desvPadAux + desv;
		}
		desvPadAux = desvPadAux * (1.0 / (comprimento.size() - 1.0));
		desvPadAux = sqrt(desvPadAux);
	}

	Point2f aux1, aux2;

	for (unsigned int k = 0; k < comprimento.size(); k++)
	{
		if ((comprimento[k] > (media - intervaloTamanho * desvPadAux)) && (comprimento[k] < (media + intervaloTamanho * desvPadAux)))
		// if((comprimento[k] > (media - intervaloTamanho)) && (comprimento[k] < (media + intervaloTamanho)))
		{
			aux1.x = reference_angulo[k].x;
			aux1.y = reference_angulo[k].y;
			reference_distancia.push_back(aux1);

			aux2.x = target_angulo[k].x;
			aux2.y = target_angulo[k].y;
			target_distancia.push_back(aux2);
		}
	}

	/*	printf("\n");
		fflush(stdout);
		for (unsigned int k = 0; k < reference_distancia.size(); k++)
		{
			printf("Xref: %5.20f, Yref: %5.20f -- Xtar: %5.20f, Ytar: %5.20f \n", reference_distancia[k].x, reference_distancia[k].y, target_distancia[k].x, target_distancia[k].y);
			fflush(stdout);
		}
		printf("\n");
		fflush(stdout);
	*/
}

int calculaNCCImagePosAbsDiff(Mat &warpImg, Mat &img_target, Mat &NCC_img_8b_porAbsDiff, Mat &abs_imgs_diff, int dimensao, double limiarAbsDiff, Mat &warpImgMask)
{
	// Criando NCC Image

	Mat NCC_img;

	NCC_img.create(img_target.size(), DataType<double>::type);

	int contador_imagem_linha;
	int contador_imagem_coluna;

	int varre_imgAux_linha;
	int varre_imgAux_coluna;

	// variaveis auxiliares para varrer imagens e preencher matrizes auxiliares de dimensao NxN
	int varre_imagem_linha;
	int varre_imagem_coluna;

	// dimensao deve ser impar, para haver um elemento central na matriz

	// verificando se dimensao e impar
	if (dimensao % 2 == 0)
		return (2);

	// verificando se dimensao e no minimo 3x3
	if (dimensao <= 1)
		return (2);

	int deslocamento = (dimensao - 1) / 2;

	// printf ("\n Deslocamento: %d \n\n", deslocamento);

	Mat auxMat_R(dimensao, dimensao, DataType<double>::type);
	Mat auxMat_T(dimensao, dimensao, DataType<double>::type);
	Mat auxMat_AbsDiff(dimensao, dimensao, DataType<double>::type);

	// printf ("\n Depois de criar matrizes auxiliares \n\n");

	// int auxcol = warpImg.cols;
	// int auxlin = warpImg.rows;

	// printf ("\n Linhas: %d , Colunas: %d \n\n", auxlin, auxcol);

	for (contador_imagem_linha = 0; contador_imagem_linha < warpImg.rows; contador_imagem_linha++)
	{
		for (contador_imagem_coluna = 0; contador_imagem_coluna < warpImg.cols; contador_imagem_coluna++)
		{

			varre_imagem_linha = contador_imagem_linha - deslocamento;
			varre_imagem_coluna = contador_imagem_coluna - deslocamento;

			for (varre_imgAux_linha = 0; varre_imgAux_linha < dimensao; varre_imgAux_linha++)
			{
				for (varre_imgAux_coluna = 0; varre_imgAux_coluna < dimensao; varre_imgAux_coluna++)
				{

					if (varre_imagem_linha < 0 || varre_imagem_coluna < 0 || varre_imagem_linha >= warpImg.rows || varre_imagem_coluna >= warpImg.cols)
					{
						auxMat_R.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = 0;
						auxMat_T.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = 0;
						auxMat_AbsDiff.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = 0;
					}
					else
					{
						auxMat_R.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = warpImg.at<unsigned char>(varre_imagem_linha, varre_imagem_coluna);
						auxMat_T.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = img_target.at<unsigned char>(varre_imagem_linha, varre_imagem_coluna);
						auxMat_AbsDiff.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = abs_imgs_diff.at<unsigned char>(varre_imagem_linha, varre_imagem_coluna);
					}
					varre_imagem_coluna++;
				}

				varre_imgAux_coluna = 0;

				varre_imagem_coluna = contador_imagem_coluna - deslocamento;
				varre_imagem_linha++;
			}

			double soma_elementos_AbsDiffAux = 0.0;

			for (int cont_matAux_lin = 0; cont_matAux_lin < auxMat_AbsDiff.rows; cont_matAux_lin++)
			{
				for (int cont_matAux_col = 0; cont_matAux_col < auxMat_AbsDiff.cols; cont_matAux_col++)
				{
					soma_elementos_AbsDiffAux = soma_elementos_AbsDiffAux + auxMat_AbsDiff.at<double>(cont_matAux_lin, cont_matAux_col);
				}
			}
			soma_elementos_AbsDiffAux = soma_elementos_AbsDiffAux / (dimensao * dimensao);

			double pixel_NCC = 127.5;

			if (soma_elementos_AbsDiffAux > limiarAbsDiff)
			{
				pixel_NCC = 0;

				// Duas matrizes auxiliares NxN formadas
				// Calcular correlacao cruzada normalizada

				double media_Raux = 0;
				double media_Taux = 0;

				double norma_Raux = 0;
				double norma_Taux = 0;

				for (int cont_matAux_lin = 0; cont_matAux_lin < auxMat_R.rows; cont_matAux_lin++)
				{
					for (int cont_matAux_col = 0; cont_matAux_col < auxMat_R.cols; cont_matAux_col++)
					{
						media_Raux = media_Raux + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col);
						media_Taux = media_Taux + auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);

						// norma_Raux = norma_Raux + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col)*auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col);
						// norma_Taux = norma_Taux + auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col)*auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);
					}
				}
				media_Raux = media_Raux / (auxMat_R.rows * auxMat_R.cols);
				media_Taux = media_Taux / (auxMat_T.rows * auxMat_T.cols);

				// double pixel_NCC = 0;

				for (int cont_matAux_lin = 0; cont_matAux_lin < auxMat_R.rows; cont_matAux_lin++)
				{
					for (int cont_matAux_col = 0; cont_matAux_col < auxMat_R.cols; cont_matAux_col++)
					{
						auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) = (auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) - media_Raux);
						auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) = (auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) - media_Taux);

						// pixel_NCC = pixel_NCC + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col)*auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);
					}
				}

				for (int cont_matAux_lin = 0; cont_matAux_lin < auxMat_R.rows; cont_matAux_lin++)
				{
					for (int cont_matAux_col = 0; cont_matAux_col < auxMat_R.cols; cont_matAux_col++)
					{
						// media_Raux = media_Raux + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col);
						// media_Taux = media_Taux + auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);

						norma_Raux = norma_Raux + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) * auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col);
						norma_Taux = norma_Taux + auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) * auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);
					}
				}

				norma_Raux = sqrt(norma_Raux);
				norma_Taux = sqrt(norma_Taux);

				for (int cont_matAux_lin = 0; cont_matAux_lin < auxMat_R.rows; cont_matAux_lin++)
				{
					for (int cont_matAux_col = 0; cont_matAux_col < auxMat_R.cols; cont_matAux_col++)
					{
						auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) = auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) / norma_Raux;
						auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) = auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col) / norma_Taux;

						pixel_NCC = pixel_NCC + auxMat_R.at<double>(cont_matAux_lin, cont_matAux_col) * auxMat_T.at<double>(cont_matAux_lin, cont_matAux_col);
					}
				}

				// if (pixel_NCC < 0)
				//   pixel_NCC = 0;

				// if (pixel_NCC > 255)
				//  pixel_NCC = 255;
			}

			// preencher elemento correspondente na NCC_img
			NCC_img.at<double>(contador_imagem_linha, contador_imagem_coluna) = pixel_NCC;
		}

		contador_imagem_coluna = 0;
	}

	// printf("\n Antes de exibir imagem NCC \n\n");

	// Mat NCC_img_8b;

	// NCC_img_8b.create(img_object.size(), img_object.type());

	// cvConvertScale (&NCC_img, &NCC_img_8b);
	for (int k = 0; k < NCC_img.rows; k++)
	{
		for (int l = 0; l < NCC_img.cols; l++)
		{
			if ((double)warpImgMask.at<unsigned char>(k, l) == 0.0)
			{
				NCC_img.at<double>(k, l) = -127.5;
			}
		}
	}

	NCC_img.convertTo(NCC_img_8b_porAbsDiff, NCC_img_8b_porAbsDiff.type(), 127.5, 127.5);

	// exibir NCC_img

	// matchTemplate(img_scene, warpImg, NCC_img, CV_TM_CCORR_NORMED);

	// imshow("NCC", NCC_img_8b);

	return (0);
}

void criaMascara(Mat &NCC_img_8b, Mat &Mask, double &threshold, Mat &warpImgMask)
{
	Mat MaskAux;
	MaskAux.create(Mask.size(), DataType<double>::type);

	double pixelAux, pixelEscrever, pixelWarp;

	for (int k = 0; k < NCC_img_8b.rows; k++)
	{
		for (int l = 0; l < NCC_img_8b.cols; l++)
		{
			pixelWarp = warpImgMask.at<unsigned char>(k, l);
			pixelAux = NCC_img_8b.at<unsigned char>(k, l);
			if (pixelWarp == 0.0)
			{
				pixelEscrever = 255.0;
			}
			else
			{
				if (pixelAux < threshold)
				{
					pixelEscrever = 0.0;
				}
				else
				{
					pixelEscrever = 255.0;
				}
			}
			MaskAux.at<double>(k, l) = pixelEscrever;
		}
	}
	MaskAux.convertTo(Mask, Mask.type(), 127.5, 127.5);
}

void criaWarpMask(Mat &warpImgMask, Mat &H)
{
	Mat matAux;
	matAux.create(warpImgMask.size(), warpImgMask.type());
	for (int k = 0; k < matAux.rows; k++)
	{
		for (int l = 0; l < matAux.cols; l++)
		{
			unsigned int intAux = 255;
			matAux.at<unsigned char>(k, l) = (unsigned char)intAux;
		}
	}
	warpPerspective(matAux, warpImgMask, H, warpImgMask.size());
}

void corrigeAbsDiffImg(Mat &abs_imgs_diff, Mat &abs_imgs_diff_warpMask, Mat &warpImgMask)
{
	abs_imgs_diff_warpMask = abs_imgs_diff;

	for (int k = 0; k < abs_imgs_diff_warpMask.rows; k++)
	{
		for (int l = 0; l < abs_imgs_diff_warpMask.cols; l++)
		{
			double pixelAux = (double)warpImgMask.at<unsigned char>(k, l);
			if (pixelAux == 0.0)
			{
				abs_imgs_diff_warpMask.at<unsigned char>(k, l) = (unsigned char)0.0;
			}
		}
	}
}

void corrigeAbsDiffImgdouble(Mat abs_imgs_diff, Mat &abs_imgs_diff_warpMask, Mat warpImgMask)
{
	abs_imgs_diff_warpMask = abs_imgs_diff;

	for (int k = 0; k < abs_imgs_diff_warpMask.rows; k++)
	{
		for (int l = 0; l < abs_imgs_diff_warpMask.cols; l++)
		{
			double pixelAux = (double)warpImgMask.at<unsigned char>(k, l);
			if (pixelAux == 0.0)
			{
				abs_imgs_diff_warpMask.at<double>(k, l) = 0.0;
			}
		}
	}
}

void calculaMaxS(double &maxS, int tamVecMaxMinS, Mat &matS)
{
	std::vector<double> maxSvec;
	std::vector<double> maxSvecAux;

	for (int k = 0; k < matS.rows; k++)
	{
		for (int l = 0; l < matS.cols; l++)
		{
			double aux1 = matS.at<double>(k, l);
			maxSvecAux.push_back(aux1);
		}
	}

	double minVecMaxAux = maxSvecAux[0];
	int posMaxAux = 0;
	double fAux = minVecMaxAux;
	maxSvec.push_back(fAux);

	for (int k = 1; k < tamVecMaxMinS; k++)
	{
		fAux = maxSvecAux[k];
		if (fAux < minVecMaxAux)
		{
			minVecMaxAux = fAux;
			posMaxAux = k;
		}

		maxSvec.push_back(fAux);
	}

	for (unsigned int k = (unsigned int)tamVecMaxMinS; k < maxSvecAux.size(); k++)
	{
		fAux = maxSvecAux[k];
		if (fAux > minVecMaxAux)
		{
			double aux2 = fAux;
			maxSvec[posMaxAux] = aux2;

			minVecMaxAux = maxSvec[0];
			posMaxAux = 0;
			for (unsigned int l = 1; l < maxSvec.size(); l++)
			{
				if (maxSvec[l] < minVecMaxAux)
				{
					minVecMaxAux = maxSvec[l];
					posMaxAux = l;
				}
			}
		}
	}

	maxS = 0.0;

	for (int k = 0; k < tamVecMaxMinS; k++)
	{
		maxS = maxS + maxSvec[k];
	}

	maxS = maxS / tamVecMaxMinS;
}

void calculaMinS(double &minS, int tamVecMaxMinS, Mat &matS)
{
	std::vector<double> minSvec;
	std::vector<double> minSvecAux;

	for (int k = 0; k < matS.rows; k++)
	{
		for (int l = 0; l < matS.cols; l++)
		{
			double aux1 = matS.at<double>(k, l);
			minSvecAux.push_back(aux1);
		}
	}

	double maxVecMinAux = minSvecAux[0];
	int posMinAux = 0;
	double fAux = maxVecMinAux;
	minSvec.push_back(fAux);

	for (int k = 1; k < tamVecMaxMinS; k++)
	{
		fAux = minSvecAux[k];
		if (fAux > maxVecMinAux)
		{
			maxVecMinAux = fAux;
			posMinAux = k;
		}

		minSvec.push_back(fAux);
	}

	for (unsigned int k = (unsigned int)tamVecMaxMinS; k < minSvecAux.size(); k++)
	{
		fAux = minSvecAux[k];
		if (fAux < maxVecMinAux)
		{
			double aux2 = fAux;
			minSvec[posMinAux] = aux2;

			maxVecMinAux = minSvec[0];
			posMinAux = 0;
			for (unsigned int l = 1; l < minSvec.size(); l++)
			{
				if (minSvec[l] > maxVecMinAux)
				{
					maxVecMinAux = minSvec[l];
					posMinAux = l;
				}
			}
		}
	}

	minS = 0.0;

	for (int k = 0; k < tamVecMaxMinS; k++)
	{
		minS = minS + minSvec[k];
	}

	minS = minS / tamVecMaxMinS;
}

void calculaCS(Mat &matS, int tamVecMaxMinS, double &CS, double comparaCS)
{
	double maxS = 0.0;
	double minS = 0.0;

	calculaMaxS(maxS, tamVecMaxMinS, matS);
	calculaMinS(minS, tamVecMaxMinS, matS);

	double valorAux = maxS - minS;

	if (valorAux > comparaCS)
	{
		CS = valorAux;
	}
	else
	{
		CS = comparaCS;
	}
}

int geraBitimg(Mat &img, int tamJanelaS, double comparaCS, int tamVecMaxMinS, double somaI, Mat &bitMat)
{
	// Mat meanMat;
	// meanMat.create(img.size(), DataType<double>::type);

	double meanS = 0.0;
	double CS = 0.0;

	int contador_imagem_linha;
	int contador_imagem_coluna;

	int varre_imgAux_linha;
	int varre_imgAux_coluna;

	// variaveis auxiliares para varrer imagens e preencher matrizes auxiliares de dimensao NxN
	int varre_imagem_linha;
	int varre_imagem_coluna;

	// dimensao deve ser impar, para haver um elemento central na matriz

	// verificando se dimensao e impar
	if (tamJanelaS % 2 == 0)
		return (2);

	// verificando se dimensao e no minimo 3x3
	if (tamJanelaS <= 1)
		return (2);

	int deslocamento = (tamJanelaS - 1) / 2;

	// printf ("\n Deslocamento: %d \n\n", deslocamento);

	Mat matS(tamJanelaS, tamJanelaS, DataType<double>::type);

	// printf ("\n Depois de criar matrizes auxiliares \n\n");

	// int auxcol = warpImg.cols;
	// int auxlin = warpImg.rows;

	// printf ("\n Linhas: %d , Colunas: %d \n\n", auxlin, auxcol);

	for (contador_imagem_linha = 0; contador_imagem_linha < img.rows; contador_imagem_linha++)
	{
		for (contador_imagem_coluna = 0; contador_imagem_coluna < img.cols; contador_imagem_coluna++)
		{

			varre_imagem_linha = contador_imagem_linha - deslocamento;
			varre_imagem_coluna = contador_imagem_coluna - deslocamento;

			for (varre_imgAux_linha = 0; varre_imgAux_linha < tamJanelaS; varre_imgAux_linha++)
			{
				for (varre_imgAux_coluna = 0; varre_imgAux_coluna < tamJanelaS; varre_imgAux_coluna++)
				{

					if (varre_imagem_linha < 0 || varre_imagem_coluna < 0 || varre_imagem_linha >= img.rows || varre_imagem_coluna >= img.cols)
					{
						matS.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = 0.0;
					}
					else
					{
						matS.at<double>(varre_imgAux_linha, varre_imgAux_coluna) = (double)img.at<unsigned char>(varre_imagem_linha, varre_imagem_coluna);
						meanS = meanS + matS.at<double>(varre_imgAux_linha, varre_imgAux_coluna);
					}
					varre_imagem_coluna++;
				}

				varre_imgAux_coluna = 0;

				varre_imagem_coluna = contador_imagem_coluna - deslocamento;
				varre_imagem_linha++;
			}

			meanS = meanS / (tamJanelaS * tamJanelaS);

			calculaCS(matS, tamVecMaxMinS, CS, comparaCS);

			double pixeli = (double)img.at<unsigned char>(contador_imagem_linha, contador_imagem_coluna);

			// double pixeliNorm = somaI + (pixeli - (meanS/CS));
			double pixeliNorm = somaI + ((pixeli - meanS) / CS);

			// printf("MeanS: %lf - CS: %lf - MeanS/CS: %lf - pixeliNorm: %lf \n", meanS, CS, meanS/CS, pixeliNorm);
			/*printf("MeanS: %lf - CS: %lf - (Pixeli-MeanS)/CS: %lf - pixeliNorm: %lf \n", meanS, CS, (pixeli - meanS)/CS, pixeliNorm);
			fflush(stdout);
			waitKey(0);*/

			if (pixeliNorm < 0.0)
			{
				pixeliNorm = 0.0;
			}

			if (pixeliNorm > 1.0)
			{
				pixeliNorm = 1.0;
			}

			bitMat.at<double>(contador_imagem_linha, contador_imagem_coluna) = pixeliNorm;

			// nao esquecer de zerar a media...
			meanS = 0.0;
		}
	}

	return (0);
}

void geraBintra(Mat &Bintra, Mat &abs_imgs_diff, double threshold3)
{
	Mat MatAux;
	MatAux.create(abs_imgs_diff.size(), DataType<double>::type);

	double pixelLido, pixelEscrever;

	for (int k = 0; k < abs_imgs_diff.rows; k++)
	{
		for (int l = 0; l < abs_imgs_diff.cols; l++)
		{
			pixelLido = abs_imgs_diff.at<double>(k, l);

			if (pixelLido > threshold3)
			{
				pixelEscrever = 0.0;
			}
			else
			{
				pixelEscrever = -127.5;
			}
			MatAux.at<double>(k, l) = pixelEscrever;
		}
	}

	MatAux.convertTo(Bintra, Bintra.type(), 127.5, 127.5);
}

void calculaMatIntersecao(Mat &intersecao, Mat &MatA, Mat &MatB)
{
	for (int k = 0; k < intersecao.rows; k++)
	{
		for (int l = 0; l < intersecao.cols; l++)
		{
			double pixelAux;
			double pixelMatA = (double)MatA.at<unsigned char>(k, l);
			double pixelMatB = (double)MatB.at<unsigned char>(k, l);
			// if(MatA.at<unsigned char>(k,l) == MatB.at<unsigned char>(k,l))
			if ((pixelMatA != 255.0) && (pixelMatB != 255.0))
			{
				// pixelAux = (double) MatA.at<unsigned char>(k,l);
				pixelAux = pixelMatA;
			}
			else
			{
				pixelAux = 255.0;
			}
			intersecao.at<unsigned char>(k, l) = (unsigned char)pixelAux;
		}
	}
}

void calculaMatSubtracao(Mat &subtracao, Mat &MatA, Mat &MatB)
{
	for (int k = 0; k < subtracao.rows; k++)
	{
		for (int l = 0; l < subtracao.cols; l++)
		{
			double pixelAux;
			double pixelMatA = (double)MatA.at<unsigned char>(k, l);
			double pixelMatB = (double)MatB.at<unsigned char>(k, l);
			// if(MatA.at<unsigned char>(k,l) == MatB.at<unsigned char>(k,l))
			if ((pixelMatA != 255.0) && (pixelMatB != 255.0))
			{
				pixelAux = 255.0;
			}
			else
			{
				// pixelAux = (double) MatA.at<unsigned char>(k,l);
				pixelAux = pixelMatA;
			}
			subtracao.at<unsigned char>(k, l) = (unsigned char)pixelAux;
		}
	}
}

void adaptaBintra(Mat &Bintra, Mat &BintraAdapt)
{
	for (int k = 0; k < Bintra.rows; k++)
	{
		for (int l = 0; l < Bintra.cols; l++)
		{
			double pixelAux = (double)Bintra.at<unsigned char>(k, l);
			if (pixelAux == 0.0)
			{
				pixelAux = 255.0;
			}
			BintraAdapt.at<unsigned char>(k, l) = (unsigned char)pixelAux;
		}
	}
}

void geraB3(Mat &B3, Mat &B2, Mat &BintraTar, Mat &BintraRef)
{
	Mat intersecao;
	intersecao.create(BintraTar.size(), BintraTar.type());

	Mat BintraTarAdapt;
	BintraTarAdapt.create(BintraTar.size(), BintraTar.type());
	Mat BintraRefAdapt;
	BintraRefAdapt.create(BintraTar.size(), BintraTar.type());

	adaptaBintra(BintraTar, BintraTarAdapt);
	adaptaBintra(BintraRef, BintraRefAdapt);

	// imshow("Adapta Bintra ref", BintraRefAdapt);
	// imshow("Adapta Bintra tar", BintraTarAdapt);
	// waitKey(0);

	// calculaMatIntersecao (intersecao, BintraTar, BintraRef);
	calculaMatIntersecao(intersecao, BintraTarAdapt, BintraRefAdapt);

	Mat subtracao;
	subtracao.create(BintraTar.size(), BintraTar.type());

	// calculaMatSubtracao(subtracao, BintraTar, intersecao);
	calculaMatSubtracao(subtracao, BintraTarAdapt, intersecao);

	calculaMatIntersecao(B3, B2, subtracao);
}

void geraVetoresEnviarCalcularHomografia(int contadorMax, double distMult, std::vector<DMatch> &good_matches, double intervaloTamanho, int soEixoX, double controlaIntervaloAngulo, unsigned int numMinimoPares, int usaTotal, int siftOrSurf, int minHessian, Mat &img_reference, std::vector<KeyPoint> &keypoints_reference, Mat &descriptors_reference, Mat &img_target, std::vector<KeyPoint> &keypoints_target, Mat &descriptors_target, std::vector<Point2f> &reference_enviar, std::vector<Point2f> &target_enviar)
{
	int contadorLoop = 0;

	while (contadorLoop < contadorMax)
	{

		detect_feature_detector(siftOrSurf, minHessian, img_reference, keypoints_reference, img_target, keypoints_target);

		detect_descriptor_extractor(siftOrSurf, img_reference, keypoints_reference, descriptors_reference, img_target, keypoints_target, descriptors_target);

		FlannBasedMatcher matcher;
		std::vector<DMatch> matches;
		matcher.match(descriptors_reference, descriptors_target, matches);

		double max_dist = 0;
		double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors_reference.rows; i++)
		{
			double dist_matches = matches[i].distance;
			if (dist_matches < min_dist)
				min_dist = dist_matches;
			if (dist_matches > max_dist)
				max_dist = dist_matches;
		}

		// printf("-- Max dist inter: %lf \n", max_dist );
		// printf("-- Min dist inter: %lf \n", min_dist );

		// printf("-- Matches : %lu \n", matches.size() );

		unsigned int numGoodMatches;

		if (usaTotal == 0)
		{
			for (int i = 0; i < descriptors_reference.rows; i++)
			{
				if (matches[i].distance < distMult * min_dist)
				{
					good_matches.push_back(matches[i]);
				}
			}

			numGoodMatches = good_matches.size();

			printf("\n Quantidade de pares de pontos gerados pelo SIFT/SURF: %d \n\n", numGoodMatches);
			fflush(stdout);
		}
		else
		{
			good_matches = matches;
			contadorLoop = contadorMax;
		}

		std::vector<Point2f> reference;
		std::vector<Point2f> target;

		for (unsigned int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			reference.push_back(keypoints_reference[good_matches[i].queryIdx].pt);
			target.push_back(keypoints_target[good_matches[i].trainIdx].pt);
		}

		reference_enviar = reference;
		target_enviar = target;

		std::vector<Point2f> reference_corrected;
		std::vector<Point2f> target_corrected;

		if (reference.size() > 1)
		{
			removePontosRepetidos(reference, target, reference_corrected, target_corrected);
			reference_enviar = reference_corrected;
			target_enviar = target_corrected;
		}

		// structAngulo inicial;

		//		std::vector<Point2f> reference_angulo;
		//		std::vector<Point2f> target_angulo;
		//
		//		if (reference_corrected.size() > 1)
		//		{
		//			double distAux = (double) img_reference.cols;
		//			removeAnguloGrande(reference_corrected, target_corrected, reference_angulo, target_angulo, controlaIntervaloAngulo, distAux);
		//			//verificaTamanho = reference_angulo.size();
		//			reference_enviar = reference_angulo;
		//			target_enviar = target_angulo;
		//		}
		//
		//		std::vector<Point2f> reference_distancia;
		//		std::vector<Point2f> target_distancia;
		//
		//		if (reference_angulo.size() > 1)
		//		{
		//			double distAux2 = (double) img_reference.cols;
		//			removeDistanciaDispar(reference_angulo, target_angulo, reference_distancia, target_distancia, distAux2, intervaloTamanho, soEixoX);
		//			//verificaTamanho = reference_distancia.size();
		//			reference_enviar = reference_distancia;
		//			target_enviar = target_distancia;
		//		}

		removeDiferentScale(keypoints_reference, keypoints_target, good_matches, good_matches);

		vector<Point2f> good_keypoints1, good_keypoints2;
		retrievePoint2fFromMatches(keypoints_reference, keypoints_target, good_matches, good_keypoints1, good_keypoints2);

		if (usaTotal == 0)
		{
			if (reference_enviar.size() < numMinimoPares)
			{
				distMult = distMult + 2.0;
			}
			else
			{
				contadorLoop = contadorMax;
				// verificaTamanho = reference_enviar.size();
			}
		}
		reference_enviar = good_keypoints1;
		target_enviar = good_keypoints2;
	}
}

void removeDiferentScale(const std::vector<KeyPoint> &keypoints_object_in, const std::vector<KeyPoint> &keypoints_scene_in, const std::vector<DMatch> matches, std::vector<DMatch> &matches_out)
{
	vector<DMatch> good_matches;
	for (unsigned int i = 0; i < matches.size(); i++)
	{
		// if(aux_matches[i].distance < 4*min_dist)
		if (keypoints_object_in[matches[i].queryIdx].octave == keypoints_scene_in[matches[i].trainIdx].octave)
			good_matches.push_back(matches[i]);
	}
	matches_out = good_matches;
}

void retrievePoint2fFromMatches(const std::vector<KeyPoint> &keypoints_object_in, const std::vector<KeyPoint> &keypoints_scene_in, const std::vector<DMatch> matches,
								std::vector<Point2f> &keypoints_object_out, std::vector<Point2f> &keypoints_scene_out)
{
	vector<Point2f> good_keypoints1, good_keypoints2;
	std::vector<int> aux_index1, aux_index2;

	for (unsigned int i = 0; i < matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		good_keypoints1.push_back(keypoints_object_in[matches[i].queryIdx].pt);
		good_keypoints2.push_back(keypoints_scene_in[matches[i].trainIdx].pt);
		aux_index1.push_back(matches[i].queryIdx);
		aux_index2.push_back(matches[i].trainIdx);
	}
	keypoints_object_out = good_keypoints1;
	keypoints_scene_out = good_keypoints2;
}

int calculaHomografia(int qualRansac, Mat &H, std::vector<Point2f> &reference_enviar, std::vector<Point2f> &target_enviar, double &multHomographyOpenCV, Mat &imgDupla)
{
	if (qualRansac == 0)
	{
		H = findHomography(reference_enviar, target_enviar, RANSAC, multHomographyOpenCV);
	}
	else
	{
		int funcionou = modifiedHomography(H, reference_enviar, target_enviar, imgDupla);
		if (funcionou != 0)
		{
			printf("Ransac modificado nao funcionou! \n\n");
			return (-1);
		}
	}

	return (0);
}

void desenhaHomografiaImgMatches(Mat &img_matches, int numColunas, int numLinhas, Mat &H)
{
	std::vector<Point2f> reference_corners(4);
	reference_corners[0] = cvPoint(0, 0);
	reference_corners[1] = cvPoint(numColunas, 0);
	reference_corners[2] = cvPoint(numColunas, numLinhas);
	reference_corners[3] = cvPoint(0, numLinhas);
	std::vector<Point2f> target_corners(4);

	perspectiveTransform(reference_corners, target_corners, H);

	line(img_matches, target_corners[0] + Point2f(numColunas, 0), target_corners[1] + Point2f(numColunas, 0), Scalar(0, 255, 0), 4);
	line(img_matches, target_corners[1] + Point2f(numColunas, 0), target_corners[2] + Point2f(numColunas, 0), Scalar(0, 255, 0), 4);
	line(img_matches, target_corners[2] + Point2f(numColunas, 0), target_corners[3] + Point2f(numColunas, 0), Scalar(0, 255, 0), 4);
	line(img_matches, target_corners[3] + Point2f(numColunas, 0), target_corners[0] + Point2f(numColunas, 0), Scalar(0, 255, 0), 4);
}

void geraB1(Mat &B1, Mat &warpImg, Mat &H, int usaAbsDiff, Mat &img_target, int dimensaoJanela, double limiarAbsDiff, double threshold)
{
	Mat warpImgMask;
	warpImgMask.create(img_target.size(), img_target.type());
	criaWarpMask(warpImgMask, H);

	// Criando NCC Image
	Mat NCC_img_8b;
	NCC_img_8b.create(img_target.size(), img_target.type());

	// Criando mascara
	Mat Mask;
	Mask.create(img_target.size(), img_target.type());

	if (usaAbsDiff == 0)
	{
		calculaNCCImage(warpImg, img_target, NCC_img_8b, dimensaoJanela, warpImgMask);
		// imshow("NCC inter", NCC_img_8b);
	}
	else
	{
		Mat abs_imgs_diff;
		absdiff(warpImg, img_target, abs_imgs_diff);
		// imshow("Abs Diff", abs_imgs_diff );
		// Mat show01;
		// resize(abs_imgs_diff, show01, Size((abs_imgs_diff.cols)*8, (abs_imgs_diff.rows)*8));
		// imshow("Abs Diff", show01 );

		Mat abs_imgs_diff_warpMask;
		abs_imgs_diff_warpMask.create(img_target.size(), img_target.type());
		corrigeAbsDiffImg(abs_imgs_diff, abs_imgs_diff_warpMask, warpImgMask);
		// imshow("Abs Diff inter", abs_imgs_diff_warpMask );
		// Mat show02;
		// resize(abs_imgs_diff_warpMask, show02, Size((abs_imgs_diff_warpMask.cols)*8, (abs_imgs_diff_warpMask.rows)*8));
		// imshow("Abs Diff inter", show02 );

		calculaNCCImagePosAbsDiff(warpImg, img_target, NCC_img_8b, abs_imgs_diff_warpMask, dimensaoJanela, limiarAbsDiff, warpImgMask);
		// Mat NCC_img_8b_show;
		// resize(NCC_img_8b, NCC_img_8b_show, Size((NCC_img_8b.cols)*8, (NCC_img_8b.rows)*8));
		// imshow("NCC pos Abs Diff inter", NCC_img_8b_show);
	}

	criaMascara(NCC_img_8b, Mask, threshold, warpImgMask);
	// Mat Mask_show;
	// resize(Mask, Mask_show, Size((Mask.cols)*8, (Mask.rows)*8));
	// imshow("Mask Pos Abs Diff inter", Mask_show);

	B1 = Mask;
}

void corrigeHinterWarpedIntraRefMask(Mat &HinterWarpedIntraRefMask, Mat &HinterWarpedIntraRefMaskCorrigida, Mat &Hinter)
{
	Mat MaskAux;
	MaskAux.create(HinterWarpedIntraRefMaskCorrigida.size(), HinterWarpedIntraRefMaskCorrigida.type());

	criaWarpMask(MaskAux, Hinter);

	HinterWarpedIntraRefMaskCorrigida = HinterWarpedIntraRefMask;

	for (int k = 0; k < HinterWarpedIntraRefMask.rows; k++)
	{
		for (int l = 0; l < HinterWarpedIntraRefMask.cols; l++)
		{
			double pixelAux;

			if (HinterWarpedIntraRefMask.at<unsigned char>(k, l) == MaskAux.at<unsigned char>(k, l))
			{
				pixelAux = 255.0;
			}
			else
			{
				pixelAux = (double)HinterWarpedIntraRefMask.at<unsigned char>(k, l);
			}

			HinterWarpedIntraRefMaskCorrigida.at<unsigned char>(k, l) = (unsigned char)pixelAux;
		}
	}
}

void geraB2(Mat &B2, Mat &HinterWarpedIntraRefMaskCorrigida, Mat &B1)
{
	Mat intersecao;
	intersecao.create(B2.size(), B2.type());

	calculaMatIntersecao(intersecao, HinterWarpedIntraRefMaskCorrigida, B1);
	// imshow("Intersecao em B2", intersecao);

	calculaMatSubtracao(B2, B1, intersecao);
}

void gerandoBintra(Mat &Bintra, double threshold, Mat &warpImg, Mat &img_target, int tamJanelaS, double comparaCS, int tamVecMaxMinS, double somaI, Mat &MaskAux)
{
	Mat bitWarp, bitTar;
	bitWarp.create(warpImg.size(), DataType<double>::type);
	bitTar.create(warpImg.size(), DataType<double>::type);

	geraBitimg(warpImg, tamJanelaS, comparaCS, tamVecMaxMinS, somaI, bitWarp);
	geraBitimg(img_target, tamJanelaS, comparaCS, tamVecMaxMinS, somaI, bitTar);

	Mat abs_imgs_diff;
	// abs_imgs_diff.create(img_reference.size(), DataType<double>::type);

	Mat bitWarp_warpMask;
	bitWarp_warpMask.create(bitWarp.size(), bitWarp.type());
	corrigeAbsDiffImgdouble(bitWarp, bitWarp_warpMask, MaskAux);

	absdiff(bitWarp_warpMask, bitTar, abs_imgs_diff);

	Mat abs_imgs_diff_warpMask;
	abs_imgs_diff_warpMask.create(abs_imgs_diff.size(), abs_imgs_diff.type());
	corrigeAbsDiffImgdouble(abs_imgs_diff, abs_imgs_diff_warpMask, MaskAux);

	geraBintra(Bintra, abs_imgs_diff_warpMask, threshold);
}

void corrigeB3tempMask(Mat &matAux, Mat &matAux_warpMask, Mat &warpMask)
{
	// matAux_warpMask = matAux;

	matAux.copyTo(matAux_warpMask);

	for (int k = 0; k < matAux_warpMask.rows; k++)
	{
		for (int l = 0; l < matAux_warpMask.cols; l++)
		{
			double pixelAux = (double)warpMask.at<unsigned char>(k, l);

			if (pixelAux == 0.0)
			{
				matAux_warpMask.at<unsigned char>(k, l) = (unsigned char)255.0;
			}
		}
	}
}

void corrigeVotingMask(Mat &matAux, Mat &matAuxcorrected, Mat &mask)
{
	// DataType<unsigned int>::type

	Mat maskAux;
	maskAux.create(Size((matAux.cols), (matAux.rows)), DataType<double>::type);
	// mask.copyTo(maskAux);

	Mat matAuxcorrectedAux;
	matAuxcorrectedAux.create(Size((matAux.cols), (matAux.rows)), DataType<double>::type);
	// matAux.copyTo(matAuxcorrectedAux);

	Mat matSub;
	matSub.create(Size((matAux.cols), (matAux.rows)), DataType<double>::type);

	matAuxcorrected.create(Size((matAux.cols), (matAux.rows)), matAux.type());

	double pixel255 = 255.0;
	double pixel0 = 0.0;
	double pixel127 = 127.5;

	for (int k = 0; k < matAux.rows; k++)
	{
		for (int l = 0; l < matAux.cols; l++)
		{
			double pixelAux1 = (double)matAux.at<unsigned char>(k, l);

			double pixelAux2 = (double)mask.at<unsigned char>(k, l);

			if (pixelAux2 == 255.0)
			{
				maskAux.at<double>(k, l) = pixel0;
			}
			else
			{
				maskAux.at<double>(k, l) = pixel255;
			}

			if (pixelAux1 == 255.0)
			{
				matAuxcorrectedAux.at<double>(k, l) = pixel0;
			}
			else
			{
				matAuxcorrectedAux.at<double>(k, l) = pixel255;
			}

			matSub.at<double>(k, l) = matAuxcorrectedAux.at<double>(k, l) - maskAux.at<double>(k, l);
		}
	}

	for (int k = 0; k < matAux.rows; k++)
	{
		for (int l = 0; l < matAux.cols; l++)
		{
			double pixelAux = matSub.at<double>(k, l);

			if (pixelAux == 255.0)
			{
				matAuxcorrected.at<unsigned char>(k, l) = (unsigned char)pixel127;
			}
			else

			{
				matAuxcorrected.at<unsigned char>(k, l) = (unsigned char)pixel255;
			}
		}
	}
}

unsigned int descobreQuadroMudancaDirecao(Mat &vetorAVarrer, double coefAngTemplate01, double coefLinearTemplate01, double coefAngTemplate02, double coefLinearTemplate02, unsigned int margemVetorAVarrer, unsigned int tamVetorAVarrer, double pontoIntersecaoX)
{

	double x, y;

	double covAtual = 0.0;
	double covMax = 0.0;

	unsigned int frameDaMudancaDeDirecao = 0;
	unsigned int frameAtual = 0;
	unsigned int frameDeMaiorCov = 0;

	unsigned int posVecVarreFrameAux = 0;
	unsigned int frameAux;

	unsigned int frameComeca;
	unsigned int frameTermina;

	frameComeca = (unsigned int)vetorAVarrer.at<double>(margemVetorAVarrer, 1);
	frameTermina = (unsigned int)vetorAVarrer.at<double>((tamVetorAVarrer - 1) - margemVetorAVarrer, 1);

	// printf("\n Dentro da funcao que calcula o quadro de mudanca de direcao. \n");
	// printf("\n frameComeca: %u \n", frameComeca);
	// printf("frameTermina: %u \n", frameTermina);
	// printf("\n");
	// fflush(stdout);
	// waitKey(0);

	Mat vetorCalc;
	vetorCalc.create(tamVetorAVarrer, 2, DataType<double>::type);

	for (frameAtual = frameComeca; frameAtual <= frameTermina; frameAtual++)
	{

		for (posVecVarreFrameAux = 0; posVecVarreFrameAux < tamVetorAVarrer; posVecVarreFrameAux++)
		{
			frameAux = (unsigned int)vetorAVarrer.at<double>(posVecVarreFrameAux, 1);
			vetorCalc.at<double>(posVecVarreFrameAux, 1) = (double)frameAux;

			x = vetorAVarrer.at<double>(posVecVarreFrameAux, 1) - (double)frameAtual + pontoIntersecaoX;

			if (frameAux <= frameAtual)
			{

				y = coefAngTemplate01 * x + coefLinearTemplate01;
				vetorCalc.at<double>(posVecVarreFrameAux, 0) = y;
			}
			else
			{
				y = coefAngTemplate02 * x + coefLinearTemplate02;
				vetorCalc.at<double>(posVecVarreFrameAux, 0) = y;
			}
		}

		// calculando cov

		double meanVetorCalc = 0.0;
		double meanVetorAVarrer = 0.0;

		for (unsigned int i = 0; i < tamVetorAVarrer; i++)
		{
			meanVetorCalc = meanVetorCalc + vetorCalc.at<double>(i, 0);
			meanVetorAVarrer = meanVetorAVarrer + vetorAVarrer.at<double>(i, 0);
		}

		meanVetorCalc = meanVetorCalc / ((double)tamVetorAVarrer);
		meanVetorAVarrer = meanVetorAVarrer / ((double)tamVetorAVarrer);

		double stdDevVetorCalc = 0.0;
		double stdDevVetorAVarrer = 0.0;

		for (unsigned int j = 0; j < tamVetorAVarrer; j++)
		{
			stdDevVetorCalc = stdDevVetorCalc + (vetorCalc.at<double>(j, 0) - meanVetorCalc) * (vetorCalc.at<double>(j, 0) - meanVetorCalc);
			stdDevVetorAVarrer = stdDevVetorAVarrer + (vetorAVarrer.at<double>(j, 0) - meanVetorAVarrer) * (vetorAVarrer.at<double>(j, 0) - meanVetorAVarrer);
		}

		stdDevVetorCalc = stdDevVetorCalc / ((double)tamVetorAVarrer);
		stdDevVetorCalc = sqrt(stdDevVetorCalc);

		stdDevVetorAVarrer = stdDevVetorAVarrer / ((double)tamVetorAVarrer);
		stdDevVetorAVarrer = sqrt(stdDevVetorAVarrer);

		double partialResult = 0.0;

		for (unsigned int k = 0; k < tamVetorAVarrer; k++)
		{
			partialResult = partialResult + (vetorAVarrer.at<double>(k, 0) - meanVetorAVarrer) * (vetorCalc.at<double>(k, 0) - meanVetorCalc);
		}

		covAtual = partialResult / (stdDevVetorCalc * stdDevVetorAVarrer * ((double)(tamVetorAVarrer - 1)));

		// printf("\n covMax: %lf \n", covMax);
		// printf("covAtual: %lf \n", covAtual);
		// printf("frameDeMaiorCov: %u \n", frameDeMaiorCov);
		// printf("frameAtual: %u \n", frameAtual);
		// fflush(stdout);

		if (covAtual > covMax)
		{
			covMax = covAtual;
			frameDeMaiorCov = frameAtual;

			// printf("\n Houve troca de frame escolhido dentro da funcao \n");
			// fflush(stdout);
		}
	}

	// waitKey(0);

	frameDaMudancaDeDirecao = frameDeMaiorCov;

	unsigned int quantoPularAux = 0;
	unsigned int quantoPular = 0;

	for (unsigned int l = 0; l < tamVetorAVarrer; l++)
	{
		if (frameDaMudancaDeDirecao == (unsigned int)vetorAVarrer.at<double>(l, 1))
		{
			quantoPularAux = l;
		}
	}

	quantoPular = tamVetorAVarrer - quantoPularAux;

	// printf("Serao pulados %u quadros \n", quantoPular);
	// fflush(stdout);

	// waitKey(0);

	// return (frameDaMudancaDeDirecao);
	return (quantoPular);
}

void qualOSentidoInicialDoMovimento(unsigned int &numFramesVerificaSentidoInicialAux, unsigned int numFramesVerificaSentidoInicial, double &verificaSentidoInicial, double deslocamentoHorizontalHomografia, int &sentidoVideoRef, unsigned int &verificaoSentidoInicialRealizada)
{
	numFramesVerificaSentidoInicialAux = numFramesVerificaSentidoInicialAux + 1;

	verificaSentidoInicial = verificaSentidoInicial + deslocamentoHorizontalHomografia;

	if (numFramesVerificaSentidoInicialAux == numFramesVerificaSentidoInicial)
	{

		if (verificaSentidoInicial > 0.0)
		{
			sentidoVideoRef = 0;
			printf("Indo da direita para a esquerda! \n");
			fflush(stdout);
		}
		else
		{
			sentidoVideoRef = 1;
			printf("Indo da esquerda para a direita! \n");
			fflush(stdout);
		}

		verificaoSentidoInicialRealizada = 1;
	}
}

int detectaPrimeiraMudancaDeSentido(double deslocamentoHorizontalHomografia, double &verificaSentidoInicial, int &sentidoVideoRef)
{

	if (verificaSentidoInicial > 0.0)
	{
		verificaSentidoInicial = verificaSentidoInicial + deslocamentoHorizontalHomografia;
		if (verificaSentidoInicial < 0.0)
		{
			// houve troca no sentido do movimento!
			sentidoVideoRef = 1;
			printf("Indo da esquerda para a direita! \n");
			fflush(stdout);
			return (1);
		}
	}
	else
	{
		verificaSentidoInicial = verificaSentidoInicial + deslocamentoHorizontalHomografia;
		if (verificaSentidoInicial > 0.0)
		{
			// houve troca no sentido do movimento!
			sentidoVideoRef = 0;
			printf("Indo da direita para a esquerda! \n");
			fflush(stdout);
			return (2);
		}
	}

	return (0);
}

int detectaDemaisMudancasDeSentido(int &varAuxImprimindo, unsigned int &emQualTroca, unsigned int &contaFrames, double &posEixoY, double deslocamentoHorizontalHomografia, unsigned int comecaAGerarVetorAVarrer, unsigned int tamVetorAVarrer, Mat &vetorAVarrer, unsigned int &contadorAuxiliarVetor, int &sentidoVideoRef, unsigned int &frameDaMudancaDeDirecao, unsigned int margemVetorAVarrer, double coefAngTemplate01EsqDir, double coefLinearTemplate01EsqDir, double coefAngTemplate02EsqDir, double coefLinearTemplate02EsqDir, double pontoIntersecaoXEsqDir, double coefAngTemplate01DirEsq, double coefLinearTemplate01DirEsq, double coefAngTemplate02DirEsq, double coefLinearTemplate02DirEsq, double pontoIntersecaoXDirEsq, cv::VideoCapture &vcapRef, const std::string videoStreamAddressRefDirEsq, const std::string videoStreamAddressRefEsqDir, Mat &imageRefColor)
{

	if (varAuxImprimindo == 0)
	{
		printf("Dentro do Template Matching - em qual troca: %u \n", emQualTroca);
		fflush(stdout);

		varAuxImprimindo = 1;
	}

	//-------------------------------------------------------------
	// printf("Dentro do primeiro Template Matching - contaFrames: %u \n", contaFrames);
	// fflush(stdout);
	//-------------------------------------------------------------

	contaFrames = contaFrames + 1;

	posEixoY = posEixoY + deslocamentoHorizontalHomografia;

	if ((contaFrames >= comecaAGerarVetorAVarrer) && (contaFrames < (comecaAGerarVetorAVarrer + tamVetorAVarrer)))
	{
		vetorAVarrer.at<double>(contadorAuxiliarVetor, 0) = posEixoY;
		vetorAVarrer.at<double>(contadorAuxiliarVetor, 1) = (double)contaFrames;

		contadorAuxiliarVetor = contadorAuxiliarVetor + 1;
	}

	//-------------------------------------------------------------
	// printf("Dentro do primeiro Template Matching - contadorAuxiliarVetor: %u \n", contadorAuxiliarVetor);
	// fflush(stdout);
	//-------------------------------------------------------------

	if (contadorAuxiliarVetor == tamVetorAVarrer)
	{
		if (sentidoVideoRef == 1)
		{
			frameDaMudancaDeDirecao = descobreQuadroMudancaDirecao(vetorAVarrer, coefAngTemplate01EsqDir, coefLinearTemplate01EsqDir, coefAngTemplate02EsqDir, coefLinearTemplate02EsqDir, margemVetorAVarrer, tamVetorAVarrer, pontoIntersecaoXEsqDir);

			// adianta o video de referencia ate esse quadro -- tem que ser o video de referencia no sentido certo do movimento
			if (!vcapRef.open(videoStreamAddressRefDirEsq))
			{
				std::cout << "Error opening reference dir-esq video" << std::endl;
				fflush(stdout);
				return (-1);
			}

			if (!vcapRef.read(imageRefColor))
			{
				std::cout << "No frame 7" << std::endl;
				return (-1);
				// cv::waitKey();
			}

			sentidoVideoRef = 0;
		}
		else
		{
			frameDaMudancaDeDirecao = descobreQuadroMudancaDirecao(vetorAVarrer, coefAngTemplate01DirEsq, coefLinearTemplate01DirEsq, coefAngTemplate02DirEsq, coefLinearTemplate02DirEsq, margemVetorAVarrer, tamVetorAVarrer, pontoIntersecaoXDirEsq);

			// adianta o video de referencia ate esse quadro -- tem que ser o video de referencia no sentido certo do movimento

			if (!vcapRef.open(videoStreamAddressRefEsqDir))
			{
				std::cout << "Error opening reference esq-dir video" << std::endl;
				fflush(stdout);
				return (-1);
			}

			if (!vcapRef.read(imageRefColor))
			{
				std::cout << "No frame 8" << std::endl;
				return (-1);
				// cv::waitKey();
			}

			sentidoVideoRef = 1;
		}

		// contaFrames recebe o valor do quadro em que se esta

		printf("\n Frame em que o target video se encontra: %u \n", contaFrames);
		fflush(stdout);

		contaFrames = frameDaMudancaDeDirecao;

		printf("Primeiro Template Matching: frame em que houve a mudanca de sentido -no vetor sendo buscado-: %u\n", frameDaMudancaDeDirecao);
		fflush(stdout);

		// zero o valor de posEixoY ? -- Acho que sim...
		posEixoY = 0.0;

		contadorAuxiliarVetor = 0;

		unsigned int contaFrameRefaux = 0;
		// Adiantando o video de referencia ate o frame correto
		printf("Adiantando o video de referencia ate o frame correto \n");
		fflush(stdout);

		while (contaFrameRefaux < frameDaMudancaDeDirecao - 1)
		{
			if (!vcapRef.read(imageRefColor))
			{
				std::cout << "No frame 9" << std::endl;

				// cv::waitKey();
			}

			contaFrameRefaux = contaFrameRefaux + 1;
		}

		// Verificar abaixo!!!!!!!!!! Devo realizar essa troca de valor da variavel emQualTroca aqui????
		emQualTroca = 3;
		varAuxImprimindo = 0;
	}

	return (0);
}

void preencheCorBranca(Mat &preencheAux)
{
	for (int k = 0; k < preencheAux.rows; k++)
	{
		for (int l = 0; l < preencheAux.cols; l++)
		{

			preencheAux.at<unsigned char>(k, l) = (unsigned char)255;
		}
	}
}

int calcula_quantidade_manchas(Mat &matOriginal, int threshold_multirresolucao)
{

	Mat operando = matOriginal.clone();

	blur(operando, operando, Size(3, 3));

	Mat threshold_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(operando, threshold_output, threshold_multirresolucao, 255, THRESH_BINARY);

	/// Find contours
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// printf("\n");
	// printf("Quantidade de contornos obtidos: %lu \n", contours.size());
	// fflush(stdout);

	int aux;

	aux = contours.size() - 1;

	return (aux);
}

void gera_mat_detect_final_multirresolucao(Mat &fusao_mascaras_voting, std::vector<Mat> &count_mask_vec, double delta_janela, std::vector<double> multiplica_area_janela_vec, std::vector<int> dimensaoJanelaNCC_vec, std::vector<unsigned int> subamostraPorNCC_vec, unsigned int num_multres, int threshold_multirresolucao, Mat img_target_big, unsigned int usaRectouAreaDaMancha, unsigned int divisorShow)
{

	FILE *fp;
	FILE *saveMorphRes;
	FILE *saveRes;

	std::vector<Mat> objs_select_vec;

	// gerando vetor de matrizes de detecção com múltiplas escalas contendo somente o conteúdo das áreas desejadas:
	for (unsigned int auxLoop; auxLoop < num_multres; auxLoop++)
	{

		Mat operando = count_mask_vec[auxLoop].clone();

		Mat aGravar_aux01, aGravar_aux02;
		aGravar_aux01 = operando.clone();
		// criando arquivos de video contendo as manchas selecionadas por resolucao:
		char nomeArquivoRes[100];

		// Comentei---------------------------------------------------------------------------------

		// sprintf(nomeArquivoRes, "/home/allan/workspace/testeVideoModulo/src/results/manchasNoSelect_res%u.yuv", auxLoop);
		// if ((saveRes = fopen(nomeArquivoRes, "a")) == NULL)
		//{
		//	printf("Arquivo de quantidade de manchas parcial antes da selecao nao pode ser aberto. \n");
		//	fflush(stdout);

		//}
		// resize(aGravar_aux01, aGravar_aux02, Size(((img_target_big.cols)), ((img_target_big.rows))));
		// uchar *dataSaveRes = aGravar_aux02.ptr();
		// fwrite(dataSaveRes, 1, aGravar_aux02.cols*aGravar_aux02.rows, saveRes);
		// fclose(saveRes);

		//----------------------------------------------------------------------------------------

		blur(operando, operando, Size(3, 3));

		Mat threshold_output;
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		/// Detect edges using Threshold
		threshold(operando, threshold_output, threshold_multirresolucao, 255, THRESH_BINARY);

		/// Find contours
		findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		/// Approximate contours to polygons + get bounding rects
		vector<vector<Point>> contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		Mat preencheAux;
		preencheAux.create(count_mask_vec[auxLoop].size(), count_mask_vec[auxLoop].type());
		preencheCorBranca(preencheAux);

		printf("\n");
		printf("Numero do elemento do vetor: %u \n", auxLoop);
		printf("Quantidade de contornos obtidos: %lu \n", contours.size());
		printf("Area da imagem subamostrada atual: %d \n", operando.rows * operando.cols);
		fflush(stdout);

		for (unsigned int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));

			double areaJanelaNCCOriginal = (double)dimensaoJanelaNCC_vec[auxLoop] * dimensaoJanelaNCC_vec[auxLoop];
			double areaJanelaNCC = ((double)subamostraPorNCC_vec[auxLoop] * subamostraPorNCC_vec[auxLoop]) * areaJanelaNCCOriginal;
			double variacao = areaJanelaNCC * delta_janela;

			double area_bouldRect = (double)boundRect[i].height * boundRect[i].width;

			double lowerbound = areaJanelaNCC - variacao;
			double upperbound = multiplica_area_janela_vec[auxLoop] * areaJanelaNCC + variacao;

			double areaDaMancha = 0.0;

			for (int k = boundRect[i].tl().x; k < boundRect[i].br().x; k++)
			{
				for (int l = boundRect[i].tl().y; l < boundRect[i].br().y; l++)
				{

					if ((unsigned int)operando.at<unsigned char>(l, k) != 255)
					{
						areaDaMancha = areaDaMancha + 1.0;
					}
				}
			}

			printf("\n");
			printf("areaJanelaNCCOriginal: %lf \n", areaJanelaNCCOriginal);
			printf("areaJanelaNCC: %lf \n", areaJanelaNCC);
			printf("area_bouldRect: %lf \n", area_bouldRect);
			printf("lowerbound: %lf \n", lowerbound);
			printf("upperbound: %lf \n", upperbound);
			printf("areaDaMancha: %lf \n", areaDaMancha);
			fflush(stdout);
			printf("\n");

			// printf("\n");
			// printf("Esperando - waitKey\n");
			// printf("\n");
			// fflush(stdout);
			// waitKey(0);

			// preencheCorBranca(preencheAux);

			double areaAusar;

			if (usaRectouAreaDaMancha == 0)
			{
				areaAusar = area_bouldRect;
			}
			else
			{
				areaAusar = areaDaMancha;
			}

			if ((areaAusar > lowerbound) && (areaAusar < upperbound))
			{
				// Mat preencheAux;
				// preencheAux.create(count_mask_vec[auxLoop].size(), count_mask_vec[auxLoop].type());

				// preencheCorBranca(preencheAux);

				printf("\n");
				printf("Condicao de area satisfeita! \n");
				printf("Outra vez - areaAusar: %lf \n", areaAusar);
				printf("Outra vez - lowerbound: %lf \n", lowerbound);
				printf("Outra vez - upperbound: %lf \n", upperbound);
				fflush(stdout);
				printf("\n");

				for (int k = boundRect[i].tl().x; k < boundRect[i].br().x; k++)
				{
					for (int l = boundRect[i].tl().y; l < boundRect[i].br().y; l++)
					{

						preencheAux.at<unsigned char>(l, k) = operando.at<unsigned char>(l, k);
						// preencheAux.at<unsigned char>(k,l) = operando.at<unsigned char>(k,l);
					}
				}

				// imshow("preencheAux", preencheAux);
				// imshow("operando", operando);
				// printf("\n");
				// printf("Esperando - waitKey\n");
				// printf("\n");
				// fflush(stdout);
				// waitKey(0);
			}

			// printf("Top left: %d -- %d\n", boundRect[i].tl().x, boundRect[i].tl().y);
			// printf("Bottom right: %d -- %d\n", boundRect[i].br().x, boundRect[i].br().y);
			// printf("Width: %d\n", boundRect[i].width);
			// printf("Heigth: %d\n", boundRect[i].height);
			// printf("Area: %d\n", boundRect[i].height * boundRect[i].width);
			// printf("\n");
			// fflush(stdout);
		}

		// imshow("preencheAux", preencheAux);
		// imshow("operando", operando);
		// fflush(stdout);
		// waitKey(1);

		objs_select_vec.push_back(preencheAux);
	}

	std::vector<Mat> objs_select_Big_vec;

	// printf("\n");
	// printf("gerando vetor de matrizes com todas agora na mesma escala \n");
	// gerando vetor de matrizes com todas agora na mesma escala:
	for (unsigned int auxLoop2 = 0; auxLoop2 < num_multres; auxLoop2++)
	{
		Mat corrigeTamAux;

		Mat AuxMat = objs_select_vec[auxLoop2].clone();

		resize(AuxMat, corrigeTamAux, Size(((img_target_big.cols)), ((img_target_big.rows))));

		// imshow("corrigeTamAux", corrigeTamAux);
		// fflush(stdout);
		// waitKey(1);

		objs_select_Big_vec.push_back(corrigeTamAux);
	}
	// printf("termino gerando vetor de matrizes com todas agora na mesma escala \n");
	// printf("\n");
	// fflush(stdout);
	// waitKey(0);

	for (unsigned int contAuxImprime = 0; contAuxImprime < num_multres; contAuxImprime++)
	{

		Mat imprimeSelected;
		resize(objs_select_Big_vec[contAuxImprime], imprimeSelected, Size(((img_target_big.cols) / (divisorShow)), ((img_target_big.rows) / (divisorShow))));

		Mat matSaveMorphRes = objs_select_Big_vec[contAuxImprime].clone();

		char strnumImprime[80];
		sprintf(strnumImprime, "Selected_Final_Mask_%u", contAuxImprime);
		// imshow(strnumImprime, imprimeSelected);

		// criando arquivos de texto contendo a quantidade de objetos por resolucao:
		char nomeArquivo[100];
		sprintf(nomeArquivo, "/home/allan.freitas/testeVideoModulo/results/quantidade_objs_res_%u", contAuxImprime);
		if ((fp = fopen(nomeArquivo, "a")) == NULL)
		{
			printf("Arquivo de quantidade de manchas parcial nao pode ser aberto. \n");
			fflush(stdout);
		}
		int qtd_manchas_aux;
		qtd_manchas_aux = calcula_quantidade_manchas(objs_select_Big_vec[contAuxImprime], threshold_multirresolucao);
		fprintf(fp, "%d\n", qtd_manchas_aux);
		fclose(fp);

		// criando arquivos de video contendo as manchas selecionadas por resolucao:
		// COMENTEI---------------------------------------------------------------------------------

		// char nomeArquivoMorphRes[100];
		// sprintf(nomeArquivoMorphRes, "/home/allan/workspace/testeVideoModulo/src/results/manchas_res%u.yuv", contAuxImprime);
		// if ((saveMorphRes = fopen(nomeArquivoMorphRes, "a")) == NULL)
		//{
		// printf("Arquivo de quantidade de manchas parcial nao pode ser aberto. \n");
		// fflush(stdout);

		//}
		// uchar *dataSaveMorphRes = matSaveMorphRes.ptr();
		// fwrite(dataSaveMorphRes, 1, matSaveMorphRes.cols*matSaveMorphRes.rows, saveMorphRes);

		// fclose(saveMorphRes);

		//----------------------------------------------------------------------------------------
	}

	fusao_mascaras_voting.create(img_target_big.size(), count_mask_vec[0].type());
	preencheCorBranca(fusao_mascaras_voting);

	// Gerando máscara com a mancha de detecão final:
	// talvez seja necessário trocar essa função por outra semelhante, que copie só o interior das áreas dos retângulos...
	for (unsigned int auxLoop3 = 0; auxLoop3 < num_multres; auxLoop3++)
	{
		for (int k = 0; k < fusao_mascaras_voting.rows; k++)
		{
			for (int l = 0; l < fusao_mascaras_voting.cols; l++)
			{
				if ((double)objs_select_Big_vec[auxLoop3].at<unsigned char>(k, l) != 255.0)
				{
					fusao_mascaras_voting.at<unsigned char>(k, l) = objs_select_Big_vec[auxLoop3].at<unsigned char>(k, l);
				}
			}
		}
	}
}
