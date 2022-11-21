/*
 * localizaDummy03.cpp
 *
 *      Author: gustavo.carvalho
 */

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d.hpp"

#include "math.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "localizaDummy03Lib.h"
#include <fstream>

using namespace cv;
using namespace std;

int main(int, char **)
{

	for (int currentVideo = 1; currentVideo < 60; currentVideo++)
	{

		time_t timeStart = time(0);

		std::cout << "Current video: " << currentVideo << std::endl;

		srand(time(NULL));

		//--------------------------------------------------------

		unsigned int subTemporal = 0;

		// onde comecar a ler o video de referencia
		// int frameInicialRef = beginRef[currentVideo - 1] - 40;
		int frameInicialRef = 1;

		// onde comecar a ler o video alvo
		// int frameInicialTar = beginTar[currentVideo - 1] - 40;
		int frameInicialTar = 1;

		// onde parar de ler o video alvo
		// int terminaVideos = beginTar[currentVideo - 1] - 40 + 250;
		int terminaVideos = 201;

		// NCC:
		double threshold1 = 190.0; // entre 0 e 255
		// test1: 190, test2: 170 (valor do script antigo), test3: 140, test4: 100, test5:210, test7,8,9,12: 230, test10,11:240
		int dimensaoJanelaInter = 5; // deve ser impar
		unsigned int subamostraPor = 8;

		// tests 1:7
		unsigned int tamVecNCC = 5;
		unsigned int tamVecTF = 16;
		unsigned int votThreshold = 7;

		int totalLinhasQuadro, totalColunasQuadro;

		// const std::string videoStreamAddressRef = videoRef[currentVideo - 1];
		std::stringstream streamnameref;
		streamnameref << "/nfs/proc/luiz.tavares/VDAO_Database/data/test/videos/ref/" << std::setw(2) << std::setfill('0') << currentVideo << ".avi"; // ALTERAR DIRETORIO DE ENTRADA DA REFERENCIA
		const std::string videoStreamAddressRef = streamnameref.str();

		std::stringstream streamnametar;
		// const std::string videoStreamAddressTar = videoTar[currentVideo - 1];
		streamnametar << "/nfs/proc/luiz.tavares/VDAO_Database/data/test/videos/tar/" << std::setw(2) << std::setfill('0') << currentVideo << ".avi"; // ALTERAR DIRETORIO DE ENTRADA DO TARGET
		const std::string videoStreamAddressTar = streamnametar.str();

		int divisorSIFTSURF = 4;
		int contadorMax = 5;
		double distMult = 2.0;
		double intervaloTamanho_inter = 0.5;
		int soEixoX_inter = 0;
		double controlaIntervaloAngulo_inter = 0.015625;

		int siftOrSurf = 0;
		int usaTotal = 1;
		int minHessian = 400;
		unsigned int numMinimoPares_inter = 40;
		unsigned int divisorShow = 4;
		int qualRansac = 0;
		double multHomographyOpenCV = 2.0;
		int usaAbsDiff = 1;
		double limiarAbsDiff_inter = 5.0;

		int scaleDependent = 1;
		int usaRestricao1grau = 0;
		int usaMediaDistancia = 1;

		int salvaQual = 2;
		int fechamento_elem = 2;
		int fechamento_size = 5;
		int abertura_elem = 2;
		int abertura_size = 4;

		unsigned int tamVecH = tamVecNCC - 1;
		if (tamVecH < tamVecTF - 1)
		{
			tamVecH = tamVecTF - 1;
		}

		std::vector<Mat> vecNCC, vecH, vecTF;

		unsigned int subTemporalAux = subTemporal;

		cv::VideoCapture vcapRef, vcapTar;

		if (!vcapRef.open(videoStreamAddressRef))
		{
			std::cout << "Error opening reference video" << std::endl;
			fflush(stdout);
			return (-1);
		}

		if (!vcapTar.open(videoStreamAddressTar))
		{
			std::cout << "Error opening target video" << std::endl;
			fflush(stdout);
			return (-1);
		}

		// vcapRef.set(CAP_PROP_FRAME_WIDTH, 1280.0);
		// vcapRef.set(CAP_PROP_FRAME_HEIGHT, 720.0);
		// vcapTar.set(CAP_PROP_FRAME_WIDTH, 1280.0);
		// vcapTar.set(CAP_PROP_FRAME_HEIGHT, 720.0);

		cv::Mat imageRefColor, imageTarColor;
		Mat imageTarColorOld;

		Mat img_reference_big, img_target_big;
		Mat img_reference, img_target;

		if (!vcapRef.read(imageRefColor))
		{
			std::cout << "No frame" << std::endl;
			return (-1);
			// cv::waitKey();
		}

		if (!vcapTar.read(imageTarColor))
		{
			std::cout << "No frame" << std::endl;
			return (-1);
			// cv::waitKey();
		}

		// --------------------------------------------------------- Alinhando videos ---------------------------------------------------------
		int contaFrameRef = 1; // o primeiro frame foi lido como teste para a abertura do video
		int contaFrameTar = 1; // o primeiro frame foi lido como teste para a abertura do video

		while (contaFrameRef < frameInicialRef)
		{
			if (!vcapRef.read(imageRefColor))
			{
				std::cout << "No reference frame" << std::endl;

				// cv::waitKey();
			}

			contaFrameRef = contaFrameRef + 1;
		}

		while (contaFrameTar < frameInicialTar)
		{
			// imageTarColorOld = imageTarColor.clone();
			if (!vcapTar.read(imageTarColor))
			{
				std::cout << "No target frame" << std::endl;

				// cv::waitKey();
			}

			contaFrameTar = contaFrameTar + 1;
		}

		Mat escalaHomografia1;
		escalaHomografia1.create(3, 3, DataType<double>::type);

		escalaHomografia1.at<double>(0, 0) = (1.0 / ((double)subamostraPor));
		escalaHomografia1.at<double>(0, 1) = 0.0;
		escalaHomografia1.at<double>(0, 2) = 0.0;
		escalaHomografia1.at<double>(1, 0) = 0.0;
		escalaHomografia1.at<double>(1, 1) = (1.0 / ((double)subamostraPor));
		escalaHomografia1.at<double>(1, 2) = 0.0;
		escalaHomografia1.at<double>(2, 0) = 0.0;
		escalaHomografia1.at<double>(2, 1) = 0.0;
		escalaHomografia1.at<double>(2, 2) = 1.0;

		Mat escalaHomografia2;
		escalaHomografia2.create(3, 3, DataType<double>::type);

		escalaHomografia2.at<double>(0, 0) = (double)subamostraPor;
		escalaHomografia2.at<double>(0, 1) = 0.0;
		escalaHomografia2.at<double>(0, 2) = 0.0;
		escalaHomografia2.at<double>(1, 0) = 0.0;
		escalaHomografia2.at<double>(1, 1) = (double)subamostraPor;
		escalaHomografia2.at<double>(1, 2) = 0.0;
		escalaHomografia2.at<double>(2, 0) = 0.0;
		escalaHomografia2.at<double>(2, 1) = 0.0;
		escalaHomografia2.at<double>(2, 2) = 1.0;

		int contaFrameTar2 = contaFrameTar;

		int ondeComeca = contaFrameTar2;

		FILE *saveIMGduplaResults;

		saveIMGduplaResults = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/DAOMC/outputs/detecting.bgr", "w"); // ALTERAR DIRETORIO DE SAIDA

		timeval tvcomecapart, tvterminapart;
		double temprodacodigo = 0.0;
		for (int k = ondeComeca; k < terminaVideos; k++)
		{
			if (subTemporalAux == subTemporal)
			{

				gettimeofday(&tvcomecapart, NULL);

				if (!vcapRef.read(imageRefColor))
				{
					std::cout << "No frame" << std::endl;
					// cv::waitKey();
				}

				imageTarColorOld = imageTarColor.clone();
				if (!vcapTar.read(imageTarColor))
				{
					std::cout << "No frame" << std::endl;
					// cv::waitKey();
				}
				contaFrameTar2 = contaFrameTar2 + 1;

				subTemporalAux = 0;

				cvtColor(imageRefColor, img_reference_big, COLOR_BGR2GRAY);
				cvtColor(imageTarColor, img_target_big, COLOR_BGR2GRAY);

				resize(img_reference_big, img_reference, Size((img_reference_big.cols) / divisorSIFTSURF, (img_reference_big.rows) / divisorSIFTSURF));
				resize(img_target_big, img_target, Size((img_target_big.cols) / divisorSIFTSURF, (img_target_big.rows) / divisorSIFTSURF));

				// normalizeImage(img_reference, img_reference, cv::Mat());
				// normalizeImage(img_target, img_target, cv::Mat());

				std::vector<Point2f> reference_enviar;
				std::vector<Point2f> target_enviar;

				std::vector<DMatch> good_matches_inter;

				int verificaTamanhoInter;

				Mat img_matches_inter;

				std::vector<KeyPoint> keypoints_reference, keypoints_target;
				Mat descriptors_reference, descriptors_target;

				Mat imgDupla_inter;
				geraImagemDupla(img_reference, img_target, imgDupla_inter);

				timeval tv1, tv2;
				gettimeofday(&tv1, NULL);
				geraVetoresEnviarCalcularHomografia(scaleDependent, usaRestricao1grau, usaMediaDistancia, contadorMax, distMult, good_matches_inter, intervaloTamanho_inter, soEixoX_inter, controlaIntervaloAngulo_inter, numMinimoPares_inter, usaTotal, siftOrSurf, minHessian, img_reference, keypoints_reference, descriptors_reference, img_target, keypoints_target, descriptors_target, reference_enviar, target_enviar);
				gettimeofday(&tv2, NULL);

				FILE *gravaSURFtemp;

				if ((gravaSURFtemp = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/DAOMC/outputs/temposSURF.txt", "a")) == NULL) // ALTERAR DIRETORIO DE SAIDA
				{
					printf("Arquivo temposSURF.txt nao pode ser aberto. \n");
					fflush(stdout);
					return (-1);
				}

				fflush(stdout);

				fprintf(gravaSURFtemp, "%lf \n", tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec) / 1e6);

				fclose(gravaSURFtemp);

				Mat Hinter;

				Mat B1;

				B1.create(Size((img_reference.cols) / subamostraPor, (img_reference.rows) / subamostraPor), img_reference.type());

				Mat mask_HintraRef;

				mask_HintraRef.create(Size((B1.cols), (B1.rows)), B1.type());

				// Mat img_matches;
				drawMatches(img_reference, keypoints_reference, img_target, keypoints_target,
							good_matches_inter, img_matches_inter, Scalar::all(-1), Scalar::all(-1),
							vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				Mat imgDupla_inter_down;
				resize(imgDupla_inter, imgDupla_inter_down, Size((imgDupla_inter.cols) * divisorSIFTSURF / divisorShow, (imgDupla_inter.rows) * divisorSIFTSURF / divisorShow));
				// cv::imshow("Duas imagens inter", imgDupla_inter_down);
				// waitKey(1);

				// printf("Tempo do SIFT/SURF inter: %lf \n", tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec) / 1e6);

				desenhando(imgDupla_inter, reference_enviar, target_enviar);
				resize(imgDupla_inter, imgDupla_inter_down, Size((imgDupla_inter.cols) / divisorShow, (imgDupla_inter.rows) / divisorShow));
				// cv::imshow("Pontos escolhidos antes do Ransac inter", imgDupla_inter_down);

				verificaTamanhoInter = reference_enviar.size();

				int numColunas = img_reference.cols;
				int numLinhas = img_reference.rows;

				printf("verificaTamanhoInter: %d \n", verificaTamanhoInter);
				fflush(stdout);

				if (verificaTamanhoInter > 3)
				{

					int funcionouHinter = calculaHomografia(qualRansac, Hinter, reference_enviar, target_enviar, multHomographyOpenCV, imgDupla_inter);

					if (funcionouHinter != 0)
					{
						printf("Nao se conseguiu gerar matriz Hinter \n");
						fflush(stdout);
						return (-1);
					}

					desenhaHomografiaImgMatches(img_matches_inter, numColunas, numLinhas, Hinter);
					Mat img_matches_inter_show;
					// resize(img_matches_inter, img_matches_inter_show, Size((img_matches_inter.cols)*divisorSIFTSURF/divisorShow, (img_matches_inter.rows)*divisorSIFTSURF/divisorShow));
					// imshow("img_matches inter", img_matches_inter);

					desenhaHomografiaImgMatches(imgDupla_inter, numColunas, numLinhas, Hinter);
					Mat imgDupla_inter_show;
					// resize(imgDupla_inter, imgDupla_inter_show, Size((imgDupla_inter.cols)*divisorSIFTSURF/divisorShow, (imgDupla_inter.rows)*divisorSIFTSURF/divisorShow));
					// imshow("imgDupla_inter", imgDupla_inter);
					// imwrite("/home/gustavo.carvalho/workspace/localizaDummy01/imgDupla_inter.jpg", imgDupla_inter);

					// waitKey(1);

					Mat warpImg_inter;
					warpImg_inter.create(img_reference.size(), img_reference.type());
					warpPerspective(img_reference, warpImg_inter, Hinter, warpImg_inter.size());

					Mat warpImg_inter_down;
					resize(warpImg_inter, warpImg_inter_down, Size((B1.cols), (B1.rows)));

					Mat img_target_down;
					resize(img_target, img_target_down, Size((B1.cols), (B1.rows)));

					Mat HinterEsc;

					HinterEsc = escalaHomografia1 * Hinter * escalaHomografia2;

					timeval tv5, tv6;
					gettimeofday(&tv5, NULL);
					geraB1(B1, warpImg_inter_down, HinterEsc, usaAbsDiff, img_target_down, dimensaoJanelaInter, limiarAbsDiff_inter, threshold1);
					gettimeofday(&tv6, NULL);

					FILE *gravaNCCtemp;

					if ((gravaNCCtemp = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/DAOMC/outputs/temposNCC.txt", "a")) == NULL) // ALTERAR DIRETORIO DE SAIDA
					{
						printf("Arquivo temposSNCC.txt nao pode ser aberto. \n");
						fflush(stdout);
						return (-1);
					}

					printf("Tempo do NCC inter: %lf \n", tv6.tv_sec - tv5.tv_sec + (tv6.tv_usec - tv5.tv_usec) / 1e6);
					fflush(stdout);

					fprintf(gravaNCCtemp, "%lf \n", tv6.tv_sec - tv5.tv_sec + (tv6.tv_usec - tv5.tv_usec) / 1e6);

					fclose(gravaNCCtemp);

					Mat B1show;
					resize(B1, B1show, Size(((B1.cols) * (subamostraPor / divisorShow)), ((B1.rows) * (subamostraPor / divisorShow))));
					// imshow("B1", B1show);

					waitKey(1);

					// Mat B1clone = B1.clone();
					Mat B1_alt_size;
					resize(B1, B1_alt_size, Size((img_target.cols), (img_target.rows)));
					// imshow("B1_alt_size", B1_alt_size);
					// waitKey(1);
					vecNCC.push_back(B1_alt_size);
					// printf("Tamanho vecNCC: %d \n", vecNCC.size());

					if (vecNCC.size() > tamVecNCC)
					{
						vecNCC.erase(vecNCC.begin());
					}
					// printf("De novo - Tamanho vecNCC: %d \n", vecNCC.size());
					// waitKey(0);
				}

				if (vecNCC.size() > 1)
				{
					Mat imgTarAtual_big, imgTarAtraso_big;

					Mat imgTarAtual, imgTarAtraso;

					cvtColor(imageTarColor, imgTarAtual_big, COLOR_BGR2GRAY);
					cvtColor(imageTarColorOld, imgTarAtraso_big, COLOR_BGR2GRAY);

					resize(imgTarAtual_big, imgTarAtual, Size(((imgTarAtual_big.cols) / divisorSIFTSURF), ((imgTarAtual_big.rows) / divisorSIFTSURF)));
					resize(imgTarAtraso_big, imgTarAtraso, Size(((imgTarAtraso_big.cols) / divisorSIFTSURF), ((imgTarAtraso_big.rows) / divisorSIFTSURF)));

					std::vector<Point2f> imgTar_enviar_atual;
					std::vector<Point2f> imgTar_enviar_atraso;
					std::vector<DMatch> good_matches_imgTar;
					std::vector<KeyPoint> keypoints_imgTar_atraso, keypoints_imgTar_atual;
					Mat descriptors_imgTar_atraso, descriptors_imgTar_atual;
					Mat img_matches_imgTar;
					int verificaTamanhoimgTar;

					Mat imgDupla_imgTar;
					geraImagemDupla(imgTarAtraso, imgTarAtual, imgDupla_imgTar);

					timeval tv3, tv4;
					gettimeofday(&tv3, NULL);
					geraVetoresEnviarCalcularHomografia(scaleDependent, usaRestricao1grau, usaMediaDistancia, contadorMax, distMult, good_matches_imgTar, intervaloTamanho_inter, soEixoX_inter, controlaIntervaloAngulo_inter, numMinimoPares_inter, usaTotal, siftOrSurf, minHessian, imgTarAtraso, keypoints_imgTar_atraso, descriptors_imgTar_atraso, imgTarAtual, keypoints_imgTar_atual, descriptors_imgTar_atual, imgTar_enviar_atraso, imgTar_enviar_atual);
					gettimeofday(&tv4, NULL);

					FILE *gravaSURFtemp2;

					if ((gravaSURFtemp2 = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/DAOMC/outputs/temposSURF2.txt", "a")) == NULL) // ALTERAR DIRETORIO DE SAIDA
					{
						printf("Arquivo temposSURF2.txt nao pode ser aberto. \n");
						fflush(stdout);
						return (-1);
					}

					printf("Tempo do SIFT/SURF intra: %lf \n", tv4.tv_sec - tv3.tv_sec + (tv4.tv_usec - tv3.tv_usec) / 1e6);
					fflush(stdout);

					fprintf(gravaSURFtemp2, "%lf \n", tv4.tv_sec - tv3.tv_sec + (tv4.tv_usec - tv3.tv_usec) / 1e6);

					fclose(gravaSURFtemp2);

					Mat Hintra;

					drawMatches(imgTarAtraso, keypoints_imgTar_atraso, imgTarAtual, keypoints_imgTar_atual,
								good_matches_imgTar, img_matches_imgTar, Scalar::all(-1), Scalar::all(-1),
								vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

					desenhando(imgDupla_imgTar, imgTar_enviar_atraso, imgTar_enviar_atual);

					verificaTamanhoimgTar = imgTar_enviar_atraso.size();

					if (verificaTamanhoimgTar > 3)
					{
						int funcionouHB3 = calculaHomografia(qualRansac, Hintra, imgTar_enviar_atraso, imgTar_enviar_atual, multHomographyOpenCV, imgDupla_imgTar);
						if (funcionouHB3 != 0)
						{
							printf("Nao se conseguiu gerar matriz HB3 \n");
							fflush(stdout);
							return (-1);
						}

						desenhaHomografiaImgMatches(img_matches_imgTar, numColunas, numLinhas, Hintra);
						// imshow("img_matches B3", img_matches_B3);

						Mat HintraAux;
						HintraAux = Hintra.clone();
						vecH.push_back(HintraAux);

						if (vecH.size() > tamVecH)
						{
							vecH.erase(vecH.begin());
						}
					}
					// else
					//{
					//  Se uma nova homografia nao foi gerada, devo apagar o ultimo elemento do vetor de NCC
					// vecNCC.erase(vecNCC.begin() + vecNCC.size() - 1);
					//}
				}

				if (vecNCC.size() == tamVecNCC)
				{

					timeval tvtempfil1, tvtempfil2;
					gettimeofday(&tvtempfil1, NULL);
					temporalFilter(vecNCC, tamVecNCC, vecH, escalaHomografia1, escalaHomografia2, vecTF, tamVecTF);
					gettimeofday(&tvtempfil2, NULL);

					FILE *gravaTempFiltemp;

					if ((gravaTempFiltemp = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/DAOMC/outputs/temposTempFil.txt", "a")) == NULL) // ALTERAR DIRETORIO DE SAIDA
					{
						printf("Arquivo temposTempFil.txt nao pode ser aberto. \n");
						fflush(stdout);
						return (-1);
					}

					printf("Tempo da filtragem temporal: %lf \n", tvtempfil2.tv_sec - tvtempfil1.tv_sec + (tvtempfil2.tv_usec - tvtempfil1.tv_usec) / 1e6);
					fflush(stdout);

					fprintf(gravaTempFiltemp, "%lf \n", tvtempfil2.tv_sec - tvtempfil1.tv_sec + (tvtempfil2.tv_usec - tvtempfil1.tv_usec) / 1e6);

					fclose(gravaTempFiltemp);
				}

				Mat votResult;
				Mat morphOpOutput;

				if (vecTF.size() == tamVecTF)
				{
					timeval tvvot1, tvvot2;
					gettimeofday(&tvvot1, NULL);
					// cout << "Here 1." << endl;

					votingStep(vecTF, tamVecTF, vecH, escalaHomografia1, escalaHomografia2, votThreshold, votResult);
					// cout << "Here 2" << endl;
					gettimeofday(&tvvot2, NULL);

					FILE *gravaVottemp;

					if ((gravaVottemp = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/DAOMC/outputs/temposVot.txt", "a")) == NULL) // ALTERAR DIRETORIO DE SAIDA
					{
						printf("Arquivo temposVot.txt nao pode ser aberto. \n");
						fflush(stdout);
						return (-1);
					}

					printf("Tempo da votacao: %lf \n", tvvot2.tv_sec - tvvot1.tv_sec + (tvvot2.tv_usec - tvvot1.tv_usec) / 1e6);
					fflush(stdout);

					fprintf(gravaVottemp, "%lf \n", tvvot2.tv_sec - tvvot1.tv_sec + (tvvot2.tv_usec - tvvot1.tv_usec) / 1e6);

					fclose(gravaVottemp);

					morphOps(salvaQual, abertura_elem, abertura_size, fechamento_elem, fechamento_size, votResult, morphOpOutput);
				}

				if (vecTF.size() == tamVecTF)
				{

					Vec3b corPonto2;
					corPonto2.val[0] = 0;
					corPonto2.val[1] = 0;
					corPonto2.val[2] = 0;

					Mat usaFim;

					resize(morphOpOutput, usaFim, Size(img_target_big.cols, img_target_big.rows));

					Mat img_targetresult;
					Mat img_targetresult_color;

					img_targetresult = img_target_big.clone();

					cvtColor(img_targetresult, img_targetresult_color, COLOR_GRAY2BGR);

					for (int k = 0; k < img_targetresult.rows; k++)
					{
						for (int l = 0; l < img_targetresult.cols; l++)
						{
							if ((double)usaFim.at<unsigned char>(k, l) != 255.0)
							{

								corPonto2.val[0] = img_targetresult_color.at<Vec3b>(k, l).val[0] / 3;
								corPonto2.val[1] = img_targetresult_color.at<Vec3b>(k, l).val[1] / 3;
								corPonto2.val[2] = img_targetresult_color.at<Vec3b>(k, l).val[2];

								img_targetresult_color.at<Vec3b>(k, l) = corPonto2;
							}
						}
					}

					// imshow("Resultado", img_targetresult_color);
					// waitKey(1);
					uchar *dataColor = img_targetresult_color.ptr();
					fwrite(dataColor, 1, img_targetresult_color.cols * img_targetresult_color.rows * 3, saveIMGduplaResults);

					totalLinhasQuadro = img_targetresult_color.rows;
					totalColunasQuadro = img_targetresult_color.cols;

					// ADICIONEI
					std::stringstream sstm;
					//				sstm << "/run/media/allan/Elements/Lucas/"<<currentVideo<<"/blob/carro/high/"<<k<<".png";
					// sstm << "/home/allan.freitas/Gustavo/" << k << ".png"; //ALTERAR DIRETORIO DE SAIDA
					sstm << "/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/DAOMC/outputs/" << currentVideo << "/" << k << ".png"; // ALTERAR DIRETORIO DE SAIDA

					string imageName = sstm.str();
					imwrite(imageName, usaFim);
					waitKey(1);
				}

				gettimeofday(&tvterminapart, NULL);
				double tempopart = tvterminapart.tv_sec - tvcomecapart.tv_sec + (tvterminapart.tv_usec - tvcomecapart.tv_usec) / 1e6;
				temprodacodigo = temprodacodigo + tempopart;
			}
			else
			{

				if (!vcapRef.read(imageRefColor))
				{
					std::cout << "No frame" << std::endl;
					// cv::waitKey();
				}

				if (!vcapTar.read(imageTarColor))
				{
					std::cout << "No frame" << std::endl;
					// cv::waitKey();
				}
				contaFrameTar2 = contaFrameTar2 + 1;

				subTemporalAux = subTemporalAux + 1;
			}
		}

		printf("Tempo total do programa em segundos: %lf \n", temprodacodigo);
		printf("Tempo total do programa em minutos: %lf \n", temprodacodigo / 60.0);
		printf("Tempo para rodar 1 frame em segundos: %lf \n", (temprodacodigo / (terminaVideos - frameInicialTar)) * (subTemporal + 1));
		fflush(stdout);

		std::ofstream myfile;
		std::stringstream streamname;
		streamname << "/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/DAOMC/outputs/" << currentVideo << "/elapsedTime.txt"; // ALTERAR DIRETORIO DE SAIDA
		string fileName = streamname.str();
		myfile.open(fileName.c_str());
		myfile << temprodacodigo << '\n';
		myfile.close();

		printf("Linhas quadro: %d\n", totalLinhasQuadro);
		printf("Colunas quadro: %d", totalColunasQuadro);
		fflush(stdout);

		fclose(saveIMGduplaResults);
	}
	return (0);
}
