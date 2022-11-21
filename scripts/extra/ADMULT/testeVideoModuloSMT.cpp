/*
 * testeVideoModulo02.cpp
 *
 *      Author: gustavo
 */

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <sstream>
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
#include <sys/time.h>
#include "testeVideoModulo02Lib.h"
#include <fstream>

using namespace cv;
using namespace std;

int main(int, char **)
{

	for (int currentVideo = 1; currentVideo < 60; currentVideo++)
	{

		//	int remakeVideo [] = {12, 13, 19, 22, 32};
		//    for(int remakeIndex = 0; remakeIndex < 5; remakeIndex++)
		//    {
		//    int currentVideo = remakeVideo[remakeIndex];

		time_t timeStart = time(0);

		std::cout << "Current video: " << currentVideo << std::endl;

		srand(time(NULL));

		// lendo arquivo de configuracao
		FILE *fp;

		// if ((fp = fopen("/home/gustavo/workspace/testeVideoModulo02/config/config_umbrella.txt", "r" )) == NULL)
		if ((fp = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/config/config_teste.txt", "r")) == NULL) // ALTERAR DIRETORIO DO ARQUIVO DE CONFIGURACAO
		{
			printf("Arquivo de configuracao nao pode ser aberto. \n");
			fflush(stdout);
			return (-1);
		}

		int lendoArquAuxInt; // criado para auxiliar na leitura dos dados do arquivo de configuracao

		// variaveis abaixo lidas do arquivo de configuracao
		int frameInicialRef;
		int frameInicialTar;
		int terminaVideos;
		unsigned int subTemporal;
		unsigned int divisorSIFTSURF;
		int contadorMax;
		double distMult;
		double intervaloTamanho_inter;
		int soEixoX_inter;
		double controlaIntervaloAngulo_inter;
		unsigned int numMinimoPares_inter;
		int usaTotal;
		int siftOrSurf;
		int minHessian;
		unsigned int divisorShow;
		unsigned int subamostraOuNaoParaExibir;
		int qualRansac;
		double multHomographyOpenCV;
		int usaAbsDiff;

		double limiarAbsDiff_inter;
		double threshold1;

		unsigned int tamVecB3;
		unsigned int qual_idp_armazenar;
		unsigned int tam_vec_idp;
		unsigned int exibeMaior;
		unsigned int num_multres;

		int threshold_multirresolucao;

		fscanf(fp, "%d - %d", &lendoArquAuxInt, &frameInicialRef);
		printf("%d - %d \n", lendoArquAuxInt, frameInicialRef);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &frameInicialTar);
		printf("%d - %d \n", lendoArquAuxInt, frameInicialTar);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &terminaVideos);
		printf("%d - %d \n", lendoArquAuxInt, terminaVideos);
		fscanf(fp, "%d - %u", &lendoArquAuxInt, &subTemporal);
		printf("%d - %u \n", lendoArquAuxInt, subTemporal);
		fscanf(fp, "%d - %u", &lendoArquAuxInt, &divisorSIFTSURF);
		printf("%d - %u \n", lendoArquAuxInt, divisorSIFTSURF);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &contadorMax);
		printf("%d - %d \n", lendoArquAuxInt, contadorMax);
		fscanf(fp, "%d - %lf", &lendoArquAuxInt, &distMult);
		printf("%d - %lf \n", lendoArquAuxInt, distMult);
		fscanf(fp, "%d - %lf", &lendoArquAuxInt, &intervaloTamanho_inter);
		printf("%d - %lf \n", lendoArquAuxInt, intervaloTamanho_inter);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &soEixoX_inter);
		printf("%d - %d \n", lendoArquAuxInt, soEixoX_inter);
		fscanf(fp, "%d - %lf", &lendoArquAuxInt, &controlaIntervaloAngulo_inter);
		printf("%d - %lf \n", lendoArquAuxInt, controlaIntervaloAngulo_inter);
		fscanf(fp, "%d - %u", &lendoArquAuxInt, &numMinimoPares_inter);
		printf("%d - %u \n", lendoArquAuxInt, numMinimoPares_inter);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &usaTotal);
		printf("%d - %d \n", lendoArquAuxInt, usaTotal);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &siftOrSurf);
		printf("%d - %d \n", lendoArquAuxInt, siftOrSurf);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &minHessian);
		printf("%d - %d \n", lendoArquAuxInt, minHessian);
		fscanf(fp, "%d - %u", &lendoArquAuxInt, &divisorShow);
		printf("%d - %u \n", lendoArquAuxInt, divisorShow);
		fscanf(fp, "%d - %u", &lendoArquAuxInt, &subamostraOuNaoParaExibir);
		printf("%d - %u \n", lendoArquAuxInt, subamostraOuNaoParaExibir);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &qualRansac);
		printf("%d - %d \n", lendoArquAuxInt, qualRansac);
		fscanf(fp, "%d - %lf", &lendoArquAuxInt, &multHomographyOpenCV);
		printf("%d - %lf \n", lendoArquAuxInt, multHomographyOpenCV);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &usaAbsDiff);
		printf("%d - %d \n", lendoArquAuxInt, usaAbsDiff);

		// fscanf(fp, "%d - %u", &lendoArquAuxInt, &subamostraPor01);
		// printf("%d - %u \n", lendoArquAuxInt, subamostraPor01);
		// fscanf(fp, "%d - %d", &lendoArquAuxInt, &dimensaoJanelaInter);
		// printf("%d - %d \n", lendoArquAuxInt, dimensaoJanelaInter);
		fscanf(fp, "%d - %lf", &lendoArquAuxInt, &limiarAbsDiff_inter);
		printf("%d - %lf \n", lendoArquAuxInt, limiarAbsDiff_inter);
		fscanf(fp, "%d - %lf", &lendoArquAuxInt, &threshold1);
		printf("%d - %lf \n", lendoArquAuxInt, threshold1);

		fscanf(fp, "%d - %u", &lendoArquAuxInt, &tamVecB3);
		printf("%d - %u \n", lendoArquAuxInt, tamVecB3);
		fscanf(fp, "%d - %u", &lendoArquAuxInt, &qual_idp_armazenar);
		printf("%d - %u \n", lendoArquAuxInt, qual_idp_armazenar);
		fscanf(fp, "%d - %u", &lendoArquAuxInt, &tam_vec_idp);
		printf("%d - %u \n", lendoArquAuxInt, tam_vec_idp);
		fscanf(fp, "%d - %u", &lendoArquAuxInt, &exibeMaior);
		printf("%d - %u \n", lendoArquAuxInt, exibeMaior);

		fscanf(fp, "%d - %u", &lendoArquAuxInt, &num_multres);
		printf("%d - %u \n", lendoArquAuxInt, num_multres);
		fscanf(fp, "%d - %d", &lendoArquAuxInt, &threshold_multirresolucao);
		printf("%d - %d \n", lendoArquAuxInt, threshold_multirresolucao);

		fclose(fp);

		//--------------------------------------------------------------------------------------------------------
		// Mudanca por causa do for introduzido
		// frameInicialRef = beginRef[currentVideo - 1] - 25;
		// frameInicialTar = beginTar[currentVideo - 1] - 25;
		// terminaVideos = beginTar[currentVideo - 1] + 201;
		frameInicialRef = 1;
		frameInicialTar = 1;
		terminaVideos = 201;
		std::cout << frameInicialRef << " ref " << frameInicialTar << " tar " << terminaVideos << " fim" << std::endl;
		//--------------------------------------------------------------------------------------------------------

		// waitKey(0);

		// fim da leitura do arquivo de configuracao

		if (exibeMaior > tam_vec_idp)
		{
			printf("Erro: exibeMaior > tam_vec_idp. \n");
			fflush(stdout);
			return (-1);
		}

		// lendo arquivo de configuracao da multirresolucao
		FILE *fp2;
		if ((fp2 = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/config/config_multires_teste.txt", "r")) == NULL) // ALTERAR DIRETORIO DO ARQUIVO DE CONFIGURACAO
		{
			printf("Arquivo de configuracao de multirresolucao nao pode ser aberto. \n");
			fflush(stdout);
			return (-1);
		}

		// o conteudo destes 3 vetores abaixo nao deve ser apagado nem modificado
		std::vector<unsigned int> subamostraPorNCC_vec;
		std::vector<int> dimensaoJanelaNCC_vec;
		std::vector<double> multiplica_area_janela_vec;

		double delta_janela;

		unsigned int usaRectouAreaDaMancha;

		unsigned int subamostraPorNCC_aux;
		int dimensaoJanelaNCC_aux;
		double multiplica_area_janela_aux;

		fscanf(fp, "%d - %lf", &lendoArquAuxInt, &delta_janela);
		printf("%d - %lf \n", lendoArquAuxInt, delta_janela);
		fscanf(fp, "%d - %u", &lendoArquAuxInt, &usaRectouAreaDaMancha);
		printf("%d - %u \n", lendoArquAuxInt, usaRectouAreaDaMancha);

		// Verificar o numero de linhas do arquivo antes!!!! Tem que ter (num_multres * 3) + 1 linhas!!!!
		// Isso por enquanto - depois devo trocar isto, fixando o tamanho da janela, e o multiplicador da area da janela, variando somente a subamostragem
		for (unsigned int contAuxMult = 0; contAuxMult < num_multres; contAuxMult++)
		{
			fscanf(fp, "%d - %u", &lendoArquAuxInt, &subamostraPorNCC_aux);
			printf("%d - %u \n", lendoArquAuxInt, subamostraPorNCC_aux);
			subamostraPorNCC_vec.push_back(subamostraPorNCC_aux);

			fscanf(fp, "%d - %d", &lendoArquAuxInt, &dimensaoJanelaNCC_aux);
			printf("%d - %d \n", lendoArquAuxInt, dimensaoJanelaNCC_aux);
			dimensaoJanelaNCC_vec.push_back(dimensaoJanelaNCC_aux);

			fscanf(fp, "%d - %lf", &lendoArquAuxInt, &multiplica_area_janela_aux);
			printf("%d - %lf \n", lendoArquAuxInt, multiplica_area_janela_aux);
			multiplica_area_janela_vec.push_back(multiplica_area_janela_aux);
		}

		fclose(fp2);
		// fim da leitura do arquivo de configuracao da multirresolucao

		// waitKey(0);

		printf("\n");
		printf("Terminada a leitura dos arquivos de configuracao. \n");
		fflush(stdout);

		// waitKey(0);

		unsigned int subTemporalAux = subTemporal;

		unsigned int qual_idp_armazenar_aux = qual_idp_armazenar;

		unsigned int saiu_result = 0;

		//	const std::string videoStreamAddressRef = "/home/allan/Downloads/ref-sing-ext-part03-video01.avi";
		//	const std::string videoStreamAddressTar = "/home/allan/Downloads/obj-sing-ext-part03-video11.avi";
		//
		//    const std::string videoStreamAddressRef = "/home/allan/Desktop/base_circuito_fechado/ref02_semluz.avi";
		//    const std::string videoStreamAddressTar = "/home/allan/Desktop/base_circuito_fechado/multiple01_semluz.avi";

		//    const std::string videoStreamAddressRef = "reference.avi";
		//    const std::string videoStreamAddressTar = "target.avi";

		// const std::string videoStreamAddressRef = "/run/media/allan/Seagate/videos_mateus/ref-sing-ext-part03-video01.avi";
		// const std::string videoStreamAddressTar = "/run/media/allan/Seagate/videos_mateus/obj-sing-ext-part03-video03.avi";

		//--------------------------------------------------------------------------------------------------------
		// Mudanca por causa do for introduzido
		// const std::string videoStreamAddressRef = videoRef[currentVideo - 1];
		// const std::string videoStreamAddressTar = videoTar[currentVideo - 1];

		std::stringstream streamnameref;
		streamnameref << "/nfs/proc/luiz.tavares/VDAO_Database/data/test/videos/ref/" << std::setw(2) << std::setfill('0') << currentVideo << ".avi";
		const std::string videoStreamAddressRef = streamnameref.str();

		std::stringstream streamnametar;
		// const std::string videoStreamAddressTar = videoTar[currentVideo - 1];
		streamnametar << "/nfs/proc/luiz.tavares/VDAO_Database/data/test/videos/tar/" << std::setw(2) << std::setfill('0') << currentVideo << ".avi";
		const std::string videoStreamAddressTar = streamnametar.str();
		//----------------------------------------------------------------------------------------------------------------------------

		cv::VideoCapture vcapRef, vcapTar;

		// open the video stream and make sure it's opened
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

		cv::Mat imageRefColor, imageTarColor;
		Mat imageTarColorOld;

		Mat img_reference_big, img_target_big;

		Mat img_reference, img_target;

		// verificar onde apagar o conteudo deste vetor (ele erá sempre uma linha, que deverá ser apagada a cada vez que ele for usado!)
		// ideia: apagar sempre antes de preenche-lo! - pensar melhor depois...
		// copiando-o para dentro do loop, não precisarei apagá-lo...
		// std::vector<Mat> NCC_vec;

		// o conteudo destes 2 vetores abaixo nao deve ser apagado nem modificado
		std::vector<Mat> escalaHomografia1_vec;
		std::vector<Mat> escalaHomografia2_vec;

		for (unsigned int contAuxEscHomMult = 0; contAuxEscHomMult < num_multres; contAuxEscHomMult++)
		{

			Mat escalaHomografia1_aux;
			escalaHomografia1_aux.create(3, 3, DataType<double>::type);

			escalaHomografia1_aux.at<double>(0, 0) = (1.0 / ((double)subamostraPorNCC_vec[contAuxEscHomMult]));
			escalaHomografia1_aux.at<double>(0, 1) = 0.0;
			escalaHomografia1_aux.at<double>(0, 2) = 0.0;
			escalaHomografia1_aux.at<double>(1, 0) = 0.0;
			escalaHomografia1_aux.at<double>(1, 1) = (1.0 / ((double)subamostraPorNCC_vec[contAuxEscHomMult]));
			escalaHomografia1_aux.at<double>(1, 2) = 0.0;
			escalaHomografia1_aux.at<double>(2, 0) = 0.0;
			escalaHomografia1_aux.at<double>(2, 1) = 0.0;
			escalaHomografia1_aux.at<double>(2, 2) = 1.0;

			escalaHomografia1_vec.push_back(escalaHomografia1_aux);

			Mat escalaHomografia2_aux;
			escalaHomografia2_aux.create(3, 3, DataType<double>::type);

			escalaHomografia2_aux.at<double>(0, 0) = (double)subamostraPorNCC_vec[contAuxEscHomMult];
			escalaHomografia2_aux.at<double>(0, 1) = 0.0;
			escalaHomografia2_aux.at<double>(0, 2) = 0.0;
			escalaHomografia2_aux.at<double>(1, 0) = 0.0;
			escalaHomografia2_aux.at<double>(1, 1) = (double)subamostraPorNCC_vec[contAuxEscHomMult];
			escalaHomografia2_aux.at<double>(1, 2) = 0.0;
			escalaHomografia2_aux.at<double>(2, 0) = 0.0;
			escalaHomografia2_aux.at<double>(2, 1) = 0.0;
			escalaHomografia2_aux.at<double>(2, 2) = 1.0;

			escalaHomografia2_vec.push_back(escalaHomografia2_aux);
		}

		// acredito que lidei corretamente com os dois vetores abaixo - verificar...
		// Para a filtragem temporal:
		std::vector<vector<Mat>> vec_de_vecB3;
		std::vector<Mat> vecB3Hom;
		// std::vector<Mat> vecB3;

		// acredito que lidei corretamente com os dois vetores abaixo - verificar...
		// Para a votacao:
		std::vector<vector<Mat>> vec_de_vec_idp;
		std::vector<Mat> vec_idpHom;
		// std::vector<Mat> vec_idp;

		// verificar onde apagar o conteudo deste vetor (ele erá sempre uma linha, que deverá ser apagada a cada vez que ele for usado!)
		// movendo-o para dentro do loop, ele não vai mais precisar ser apagado...
		// std::vector<Mat> Idp_vec;
		// Mat Idp;

		Mat img_target_count, img_target_count_atraso;

		unsigned int conta_idp_armazena = 0;

		Mat count_matrix;
		Mat count_mask;

		// verificar onde apagar o conteudo deste vetor (ele erá sempre uma linha, que deverá ser apagada a cada vez que ele for usado!)
		//  ideia: apagá-lo logo após gerar a imagem com a "mancha de detecção final" - pensar melhor depois...
		// não preciso apagar este vetor caso eu o crie só quando eu for usá-lo...
		// std::vector<Mat> count_mask_vec;

		//----------------------------------------------- para exibir os resultados

		// Mat count_maskshow;
		Mat count_maskresult;
		Mat img_targetresult;
		Mat img_targetresult_show;

		Mat img_targetresult_color;
		Mat img_targetresult_color_show;

		//---------------------------------------------------------------------------

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
			imageTarColorOld = imageTarColor.clone();
			if (!vcapTar.read(imageTarColor))
			{
				std::cout << "No target frame" << std::endl;

				// cv::waitKey();
			}

			contaFrameTar = contaFrameTar + 1;
		}

		int contaFrameTar2 = contaFrameTar;

		// ------ salvando resultado em video -------

		FILE *saveIMGduplaResults;

		saveIMGduplaResults = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/detecting.yuv", "w"); // ALTERAR DIRETORIO DA SAIDA

		// ------ salvando exemplo de multirresolucao em video (so para 3 resolucoes!!!!!!!!!!!!!!!!!!!!!!!!1111) -------

		// FILE *save3resolucoes;

		// save3resolucoes = fopen ("3resolucoes.yuv", "w");

		FILE *saveMorph;
		saveMorph = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/saveMorph.yuv", "w"); // ALTERAR DIRETORIO DA SAIDA

		FILE *save_quantidade_manchas_total;

		// para salvar a quantidade de falsos positivos e negativos por frame:
		// FILE *savePosNegFile;

		// if ((save_quantidade_manchas_total = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/quantidade_manchas_total.txt", "w" )) == NULL)
		//{
		// printf("Arquivo de quantidade de manchas total nao pode ser aberto. \n");
		// fflush(stdout);
		// return (-1);
		//}

		// --------------------------------------------------------- Videos alinhados ---------------------------------------------------------

		int ondeComeca = contaFrameTar2;

		for (int k = ondeComeca; k < terminaVideos; k++)
		{

			if (subTemporalAux == subTemporal)
			{
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

				std::vector<Mat> NCC_vec;

				// imshow("alinha ref", imageRefColor);
				// imshow("alinha tar", imageTarColor);
				// waitKey(0);

				printf("Frame: %d \n", contaFrameTar2);
				fflush(stdout);

				cvtColor(imageRefColor, img_reference_big, COLOR_BGR2GRAY);
				cvtColor(imageTarColor, img_target_big, COLOR_BGR2GRAY);

				resize(img_reference_big, img_reference, Size((img_reference_big.cols) / divisorSIFTSURF, (img_reference_big.rows) / divisorSIFTSURF));
				resize(img_target_big, img_target, Size((img_target_big.cols) / divisorSIFTSURF, (img_target_big.rows) / divisorSIFTSURF));

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
				geraVetoresEnviarCalcularHomografia(contadorMax, distMult, good_matches_inter, intervaloTamanho_inter, soEixoX_inter, controlaIntervaloAngulo_inter, numMinimoPares_inter, usaTotal, siftOrSurf, minHessian, img_reference, keypoints_reference, descriptors_reference, img_target, keypoints_target, descriptors_target, reference_enviar, target_enviar);
				gettimeofday(&tv2, NULL);

				// printf("Tempo do SIFT/SURF inter: %lf \n", tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec)/1e6);
				fflush(stdout);

				// printf("\n Quantidade de pares de pontos gerados enviados para o Ransac inter: %lu \n\n", reference_enviar.size());
				fflush(stdout);
				// waitKey(0);

				drawMatches(img_reference, keypoints_reference, img_target, keypoints_target,
							good_matches_inter, img_matches_inter, Scalar::all(-1), Scalar::all(-1),
							vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				if (subamostraOuNaoParaExibir == 1)
				{
					Mat imgDupla_inter_down;
					resize(imgDupla_inter, imgDupla_inter_down, Size((imgDupla_inter.cols) * divisorSIFTSURF / divisorShow, (imgDupla_inter.rows) * divisorSIFTSURF / divisorShow));
					// cv::imshow("Duas imagens inter", imgDupla_inter_down);
				}
				else
				{
					// cv::imshow("Duas imagens inter", imgDupla_inter);
				}
				// waitKey(1);

				desenhando(imgDupla_inter, reference_enviar, target_enviar);

				verificaTamanhoInter = reference_enviar.size();

				int numColunas = img_reference.cols;
				int numLinhas = img_reference.rows;

				Mat Hinter;

				// ----------------------------------------para a criacao da imagem NCC -----------------------------------
				// acredito não precisar apagar o vetor abaixo, assim como não apago os vetores de keypoints - pensar melhor depois...
				// testei no código de Testes, e aparentemente é isso mesmo...
				std::vector<Mat> B1_vec;

				for (unsigned int contAuxB1 = 0; contAuxB1 < num_multres; contAuxB1++)
				{
					Mat B1_aux;
					B1_aux.create(Size((img_reference.cols) / subamostraPorNCC_vec[contAuxB1], (img_reference.rows) / subamostraPorNCC_vec[contAuxB1]), img_reference.type());
					B1_vec.push_back(B1_aux);

					// printf("\n");
					// printf("Valor de subamostraPorNCC_vec na posicao atual: %u \n", subamostraPorNCC_vec[contAuxB1]);
					// printf("Valor da area de B1 atual: %d \n", B1_vec[contAuxB1].rows*B1_vec[contAuxB1].cols);
					fflush(stdout);
				}

				// waitKey(0);

				// ----------------------------------------para a criacao da imagem NCC -----------------------------------

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
					resize(img_matches_inter, img_matches_inter_show, Size((img_matches_inter.cols) * divisorSIFTSURF / divisorShow, (img_matches_inter.rows) * divisorSIFTSURF / divisorShow));
					// imshow("img_matches inter", img_matches_inter_show);

					Mat warpImg_inter;
					warpImg_inter.create(img_reference.size(), img_reference.type());
					warpPerspective(img_reference, warpImg_inter, Hinter, warpImg_inter.size());

					// std::vector<Mat> HinterEsc_vec;
					// std::vector<Mat> warpImg_inter_down_vec;
					// std::vector<Mat> img_target_down_vec;

					for (unsigned int contAuxB1Mult = 0; contAuxB1Mult < num_multres; contAuxB1Mult++)
					{
						Mat HinterEsc_aux;
						HinterEsc_aux = escalaHomografia1_vec[contAuxB1Mult] * Hinter * escalaHomografia2_vec[contAuxB1Mult];
						// HinterEsc_vec.push_back(HinterEsc_aux);

						// Calculando a imagem NCC-----------------------------------------------------------
						Mat warpImg_inter_down_aux;
						resize(warpImg_inter, warpImg_inter_down_aux, Size((B1_vec[contAuxB1Mult].cols), (B1_vec[contAuxB1Mult].rows)));
						// warpImg_inter_down_vec.push_back(warpImg_inter_down_aux);

						Mat img_target_down_aux;
						resize(img_target, img_target_down_aux, Size((B1_vec[contAuxB1Mult].cols), (B1_vec[contAuxB1Mult].rows)));
						// img_target_down_vec.push_back(img_target_down_aux);

						timeval tv5, tv6;
						gettimeofday(&tv5, NULL);
						geraB1(B1_vec[contAuxB1Mult], warpImg_inter_down_aux, HinterEsc_aux, usaAbsDiff, img_target_down_aux, dimensaoJanelaNCC_vec[contAuxB1Mult], limiarAbsDiff_inter, threshold1);
						gettimeofday(&tv6, NULL);

						// printf("Tempo do NCC inter: %lf \n", tv6.tv_sec - tv5.tv_sec + (tv6.tv_usec - tv5.tv_usec)/1e6);
						// fflush(stdout);

						if (subamostraOuNaoParaExibir == 1)
						{
							Mat B1show;
							resize(B1_vec[contAuxB1Mult], B1show, Size(((B1_vec[contAuxB1Mult].cols) * (subamostraPorNCC_vec[contAuxB1Mult] * divisorSIFTSURF / divisorShow)), ((B1_vec[contAuxB1Mult].rows) * (subamostraPorNCC_vec[contAuxB1Mult] * divisorSIFTSURF / divisorShow))));

							char strnumB1[80];
							sprintf(strnumB1, "B1_%u", contAuxB1Mult);
							// imshow(strnumB1, B1show);
							string nome_end_gravar_01 = "/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/"; // ALTERAR DIRETORIO DA SAIDA
							string nomeGravar = nome_end_gravar_01 + strnumB1 + "mask.jpg";
							// imwrite(nomeGravar, B1show);

							// imshow("B1", B1show);
							imwrite("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/B1mask.jpg", B1show);
						}
						else
						{
							char strnumB1[80];
							sprintf(strnumB1, "B1_%u", contAuxB1Mult);
							// imshow(strnumB1, B1_vec[contAuxB1Mult]);
							string nome_end_gravar_01 = "/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/"; // ALTERAR DIRETORIO DA SAIDA
							string nomeGravar = nome_end_gravar_01 + strnumB1 + "mask.jpg";
							// imwrite(nomeGravar, B1_vec[contAuxB1Mult]);
						}

						Mat NCCsmall = B1_vec[contAuxB1Mult].clone();
						Mat NCC_temp_aux;
						// resize(NCCsmall, NCC_temp_aux, Size(((B1_vec[contAuxB1Mult].cols)*(subamostraPorNCC_vec[contAuxB1Mult])), ((B1_vec[contAuxB1Mult].rows)*(subamostraPorNCC_vec[contAuxB1Mult]))));
						resize(NCCsmall, NCC_temp_aux, Size((img_target.cols), (img_target.rows)));
						NCC_vec.push_back(NCC_temp_aux);

						// printf("\n");
						// printf("Valor de subamostraPorNCC_vec na posicao atual: %u \n", subamostraPorNCC_vec[contAuxB1Mult]);
						// printf("Valor da area de NCC_vec atual: %d \n", NCC_vec[contAuxB1Mult].rows*NCC_vec[contAuxB1Mult].cols);
						// fflush(stdout);
						// img_target
					}

					// printf("\n");
					// printf("Esperando - waitKey\n");
					// printf("\n");
					// fflush(stdout);
					// waitKey(1);

					// Calculando a imagem NCC-----------------------------------------------------------

					// acredito não precisar apagar o vetor abaixo, assim como não apago os vetores de keypoints...
					// além disso, tem a questão da cópia de um vetor para o outro...
					//  testar depois no programa de teste se esta cópia já resolve o problema, neste caso...
					std::vector<Mat> B3Aux_vec;
					B3Aux_vec = NCC_vec;
					// antigo vec_B3 se tornou vetor de vetores de matrizes na multirresolucao
					vec_de_vecB3.push_back(B3Aux_vec);

					if (vec_de_vecB3.size() > 1)
					{
						Mat B3atual_big, B3atraso_big;

						Mat B3atual, B3atraso;

						cvtColor(imageTarColor, B3atual_big, COLOR_BGR2GRAY);
						cvtColor(imageTarColorOld, B3atraso_big, COLOR_BGR2GRAY);

						resize(B3atual_big, B3atual, Size(((B3atual_big.cols) / divisorSIFTSURF), ((B3atual_big.rows) / divisorSIFTSURF)));
						resize(B3atraso_big, B3atraso, Size(((B3atraso_big.cols) / divisorSIFTSURF), ((B3atraso_big.rows) / divisorSIFTSURF)));

						// imshow("B3atual", B3atual);
						// imshow("B3atraso", B3atraso);
						// waitKey(0);

						std::vector<Point2f> B3_enviar_atual;
						std::vector<Point2f> B3_enviar_atraso;
						std::vector<DMatch> good_matches_B3;
						std::vector<KeyPoint> keypoints_B3_atraso, keypoints_B3_atual;
						Mat descriptors_B3_atraso, descriptors_B3_atual;
						Mat img_matches_B3;
						int verificaTamanhoB3;

						Mat imgDupla_B3;
						geraImagemDupla(B3atraso, B3atual, imgDupla_B3);

						geraVetoresEnviarCalcularHomografia(contadorMax, distMult, good_matches_B3, intervaloTamanho_inter, soEixoX_inter, controlaIntervaloAngulo_inter, numMinimoPares_inter, usaTotal, siftOrSurf, minHessian, B3atraso, keypoints_B3_atraso, descriptors_B3_atraso, B3atual, keypoints_B3_atual, descriptors_B3_atual, B3_enviar_atraso, B3_enviar_atual);

						// distMult = dMaux;

						// contadorMax = contMaxAux;

						Mat HB3;

						drawMatches(B3atraso, keypoints_B3_atraso, B3atual, keypoints_B3_atual,
									good_matches_B3, img_matches_B3, Scalar::all(-1), Scalar::all(-1),
									vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

						desenhando(imgDupla_B3, B3_enviar_atraso, B3_enviar_atual);

						verificaTamanhoB3 = B3_enviar_atraso.size();

						if (verificaTamanhoB3 > 3)
						{
							int funcionouHB3 = calculaHomografia(qualRansac, HB3, B3_enviar_atraso, B3_enviar_atual, multHomographyOpenCV, imgDupla_B3);
							if (funcionouHB3 != 0)
							{
								printf("Nao se conseguiu gerar matriz HB3 \n");
								fflush(stdout);
								return (-1);
							}

							desenhaHomografiaImgMatches(img_matches_B3, numColunas, numLinhas, HB3);
							// imshow("img_matches B3", img_matches_B3);

							Mat HB3aux;
							HB3aux = HB3.clone();
							vecB3Hom.push_back(HB3aux);
						}
						else
						{
							// Se uma nova homografia nao foi gerada, devo apagar o ultimo vetor de B3 colocado no vetor de vetores
							vec_de_vecB3.erase(vec_de_vecB3.begin() + vec_de_vecB3.size() - 1);
						}
					}

					if (vec_de_vecB3.size() == tamVecB3)
					{
						// acredito que o Idp_vec deva ser criado abaixo:
						std::vector<Mat> Idp_vec;
						for (unsigned int contAuxVecB3Mult = 0; contAuxVecB3Mult < num_multres; contAuxVecB3Mult++)
						{
							Mat IdpAux1 = vec_de_vecB3[vec_de_vecB3.size() - 1][contAuxVecB3Mult].clone();

							for (unsigned int u = 0; u < (vec_de_vecB3.size()); u++)
							{
								Mat M1 = vec_de_vecB3[u][contAuxVecB3Mult].clone();

								for (unsigned int v = u; v < (vec_de_vecB3.size() - 1); v++)
								{

									// corrigido: nao precisa mais escalar nada, por conta das varias imagens NCC a serem geradas
									Mat HB3temp = vecB3Hom[v].clone();

									Mat M2;
									M2.create(NCC_vec[contAuxVecB3Mult].size(), NCC_vec[contAuxVecB3Mult].type());
									warpPerspective(M1, M2, HB3temp, NCC_vec[contAuxVecB3Mult].size());

									Mat MaskAuxHB3temp;
									MaskAuxHB3temp.create(NCC_vec[contAuxVecB3Mult].size(), NCC_vec[contAuxVecB3Mult].type());
									criaWarpMask(MaskAuxHB3temp, HB3temp);

									Mat M2warpMask;
									corrigeVotingMask(M2, M2warpMask, MaskAuxHB3temp);

									M2warpMask.copyTo(M1);
								}

								Mat intersecIdp;
								intersecIdp.create(NCC_vec[contAuxVecB3Mult].size(), NCC_vec[contAuxVecB3Mult].type());

								calculaMatIntersecao(intersecIdp, IdpAux1, M1);

								intersecIdp.copyTo(IdpAux1);
							}

							// para multirresolucao, Idp vai ter que ser vetor de matrizes, uma para cada resolucao
							Mat IdpAux2 = IdpAux1.clone();
							Idp_vec.push_back(IdpAux2);
							Mat IdpAux3 = IdpAux1.clone();
							// imshow("Idp", Idp);

							Mat Idpshow;
							// resize(Idp, Idpshow, Size(((Idp.cols)*(subamostraPor01/divisorShow)), ((Idp.rows)*(subamostraPor01/divisorShow))));
							resize(IdpAux3, Idpshow, Size(((img_reference_big.cols) / (divisorShow)), ((img_reference_big.rows) / (divisorShow))));

							char strnumIdp[80];
							sprintf(strnumIdp, "Idp_%u", contAuxVecB3Mult);
							// imshow(strnumIdp, Idpshow);
							string nome_end_gravar_01 = "/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/"; // ALTERAR DIRETORIO DA SAIDA
							string nomeGravarIdp = nome_end_gravar_01 + strnumIdp + "show.jpg";
							// imwrite(nomeGravarIdp, Idpshow);
							string nomeGravarIdpBig = nome_end_gravar_01 + strnumIdp + ".jpg";
							// imwrite(nomeGravarIdpBig, Idp_vec[contAuxVecB3Mult]);

							// waitKey(1);
						}

						// printf("\n");
						// printf("Esperando - waitKey\n");
						// printf("\n");
						// fflush(stdout);
						// waitKey(1);

						vec_de_vecB3.erase(vec_de_vecB3.begin());
						vecB3Hom.erase(vecB3Hom.begin());

						if (qual_idp_armazenar_aux == qual_idp_armazenar)
						{

							img_target_count = img_target.clone();

							// acredito não precisar apagar o vetor abaixo, assim como não apago os vetores de keypoints...
							// além disso, tem a questão da cópia de um vetor para o outro...
							//  testar depois no programa de teste se esta cópia já resolve o problema, neste caso...
							std::vector<Mat> armazena_Idp_aux_vec = Idp_vec;
							// vec_idp se torna vetor de vetores de matrizes na multirresolucao
							vec_de_vec_idp.push_back(armazena_Idp_aux_vec);

							conta_idp_armazena = conta_idp_armazena + 1;

							if (conta_idp_armazena > 1)
							{

								std::vector<Point2f> target_count_enviar;
								std::vector<Point2f> target_atraso_count_enviar;
								std::vector<DMatch> good_matches_voting;
								int verificaTamanhoVoting;
								Mat img_matches_voting;

								std::vector<KeyPoint> keypoints_target_count, keypoints_target_atraso_count;
								Mat descriptors_target_count, descriptors_target_atraso_count;

								Mat imgDupla_voting;
								geraImagemDupla(img_target_count_atraso, img_target_count, imgDupla_voting);

								geraVetoresEnviarCalcularHomografia(contadorMax, distMult, good_matches_voting, intervaloTamanho_inter, soEixoX_inter, controlaIntervaloAngulo_inter, numMinimoPares_inter, usaTotal, siftOrSurf, minHessian, img_target_count_atraso, keypoints_target_atraso_count, descriptors_target_atraso_count, img_target_count, keypoints_target_count, descriptors_target_count, target_atraso_count_enviar, target_count_enviar);

								// printf("\n Quantidade de pares de pontos gerados enviados para o Ransac voting: %lu \n\n", target_atraso_count_enviar.size());
								// fflush(stdout);

								Mat Hvoting;

								// Mat img_matches;
								drawMatches(img_target_count_atraso, keypoints_target_atraso_count, img_target_count, keypoints_target_count,
											good_matches_voting, img_matches_voting, Scalar::all(-1), Scalar::all(-1),
											vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

								//        					Mat  imgDupla_voting_down;
								//        					desenhando(imgDupla_voting, target_atraso_count_enviar, target_count_enviar);
								//        					resize(imgDupla_voting, imgDupla_voting_down, Size((imgDupla_voting.cols)/divisorShow, (imgDupla_voting.rows)/divisorShow));
								//        					cv::imshow("Pontos escolhidos antes do Ransac voting", imgDupla_voting);

								verificaTamanhoVoting = target_atraso_count_enviar.size();

								int numColunas = img_reference.cols;
								int numLinhas = img_reference.rows;

								if (verificaTamanhoVoting > 3)
								{
									int funcionouHvoting = calculaHomografia(qualRansac, Hvoting, target_atraso_count_enviar, target_count_enviar, multHomographyOpenCV, imgDupla_voting);
									if (funcionouHvoting != 0)
									{
										printf("Nao se conseguiu gerar matriz Hvoting \n");
										fflush(stdout);
										return (-1);
									}

									desenhaHomografiaImgMatches(img_matches_voting, numColunas, numLinhas, Hvoting);
									Mat img_matches_voting_show;
									resize(img_matches_voting, img_matches_voting_show, Size((img_matches_voting.cols) * divisorSIFTSURF / divisorShow, (img_matches_voting.rows) * divisorSIFTSURF / divisorShow));
									// imshow("img_matches voting", img_matches_voting_show);

									Mat Hvotingaux;
									Hvotingaux = Hvoting.clone();
									vec_idpHom.push_back(Hvotingaux);
								}
							}

							if (conta_idp_armazena == tam_vec_idp)
							{

								// criando este vetor aqui, não preciso apagá-lo mais tarde...
								std::vector<Mat> count_mask_vec;
								for (unsigned int contAuxVecVotingMult = 0; contAuxVecVotingMult < num_multres; contAuxVecVotingMult++)
								{
									count_matrix.create(Size((NCC_vec[contAuxVecVotingMult].cols), (NCC_vec[contAuxVecVotingMult].rows)), CV_8U);
									count_mask.create(Size((NCC_vec[contAuxVecVotingMult].cols), (NCC_vec[contAuxVecVotingMult].rows)), DataType<unsigned char>::type);
									for (int k = 0; k < count_matrix.rows; k++)
									{
										for (int l = 0; l < count_matrix.cols; l++)
										{
											count_matrix.at<unsigned int>(k, l) = 0;
											count_mask.at<unsigned char>(k, l) = (unsigned char)255.0;
										}
									}

									for (unsigned int u = 0; u < (vec_de_vec_idp.size()); u++)
									{
										Mat M1 = vec_de_vec_idp[u][contAuxVecVotingMult].clone();

										for (unsigned int v = u; v < (vec_de_vec_idp.size() - 1); v++)
										{

											// Por conta do somatorio de imagens NCC, nao é mais necessário escalar homografias aqui!
											Mat Hidptemp = vec_idpHom[v].clone();

											Mat M2;
											M2.create(NCC_vec[contAuxVecVotingMult].size(), NCC_vec[contAuxVecVotingMult].type());
											warpPerspective(M1, M2, Hidptemp, NCC_vec[contAuxVecVotingMult].size());

											Mat MaskAuxHidptemp;
											MaskAuxHidptemp.create(NCC_vec[contAuxVecVotingMult].size(), NCC_vec[contAuxVecVotingMult].type());
											criaWarpMask(MaskAuxHidptemp, Hidptemp);

											Mat M2warpMask;

											corrigeVotingMask(M2, M2warpMask, MaskAuxHidptemp);

											M2warpMask.copyTo(M1);
										}

										for (int k = 0; k < count_matrix.rows; k++)
										{
											for (int l = 0; l < count_matrix.cols; l++)
											{
												double comparaVotingAux = (double)M1.at<unsigned char>(k, l);

												if ((comparaVotingAux != 255.0) && (comparaVotingAux != 0.0))
												{
													count_matrix.at<unsigned int>(k, l) = count_matrix.at<unsigned int>(k, l) + 1;
												}
											}
										}
									}

									for (int k = 0; k < count_matrix.rows; k++)
									{
										for (int l = 0; l < count_matrix.cols; l++)
										{
											if (count_matrix.at<unsigned int>(k, l) >= exibeMaior)
											{
												count_mask.at<unsigned char>(k, l) = (unsigned char)127.5;
											}
										}
									}

									Mat count_maskClone = count_mask.clone();
									count_mask_vec.push_back(count_maskClone);
									Mat count_maskClone2 = count_mask.clone();

									Mat count_maskshow;
									resize(count_maskClone2, count_maskshow, Size(((img_reference_big.cols) / (divisorShow)), ((img_reference_big.rows) / (divisorShow))));

									char strnumVoting[80];
									sprintf(strnumVoting, "Voting_Mask_%u", contAuxVecVotingMult);
									// imshow(strnumVoting, count_maskshow);
									string nome_end_gravar_02 = "/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/"; // ALTERAR DIRETORIO DA SAIDA
									// ALTERAR DIRETORIO DA SAIDA

									string nomeGravarVoting = nome_end_gravar_02 + strnumVoting + "show.jpg";
									// imwrite(nomeGravarVoting, count_maskshow);
									string nomeGravarVotingBig = nome_end_gravar_02 + strnumVoting + ".jpg";
									// imwrite(nomeGravarVotingBig, count_mask_vec[contAuxVecVotingMult]);

									// ADICIONEI PARA SALVAR IMAGENS NO PAPER DO MATEUS
									std::stringstream sstm;

									// MUDAR PARA SALVAR O OUTPUT LOCALMENTE (PARA EVITAR PROBLEMAS NO HD...)
									//--------------------------------------------------------------------------------------------------------
									// Mudanca por causa do for introduzido
									sstm << "/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/outputs/" << currentVideo << "/" << k << ".png"; // ALTERAR DIRETORIO DA SAIDA
									// ALTERAR DIRETORIO DA SAIDA

									string imageName = sstm.str();
									imwrite(imageName, count_maskshow);
								}
								// printf("\n");
								// printf("Esperando - waitKey\n");
								// printf("\n");
								// fflush(stdout);
								// waitKey(0);

								vec_de_vec_idp.erase(vec_de_vec_idp.begin());
								vec_idpHom.erase(vec_idpHom.begin());
								conta_idp_armazena = conta_idp_armazena - 1;

								saiu_result = 1;

								Mat img_target_save;
								resize(img_target, img_target_save, Size(((img_target.cols) * divisorSIFTSURF / divisorShow), ((img_target.rows) * divisorSIFTSURF / divisorShow)));

								Mat fusao_mascaras_voting;

								if (count_mask_vec.size() != num_multres)
								{
									printf("\n");
									printf("TUDO ERRADO! - Tamanho do vetor com mascaras parciais diferente do de num_multres!!! \n");
									printf("\n");
									fflush(stdout);
									return (1);
								}
								cout << "Aqui1" << endl;

								gera_mat_detect_final_multirresolucao(fusao_mascaras_voting, count_mask_vec, delta_janela, multiplica_area_janela_vec, dimensaoJanelaNCC_vec, subamostraPorNCC_vec, num_multres, threshold_multirresolucao, img_target_big, usaRectouAreaDaMancha, divisorShow);

								Mat fusao_mascaras_voting_show;

								resize(fusao_mascaras_voting, fusao_mascaras_voting_show, Size(((img_reference_big.cols) / divisorShow), ((img_reference_big.rows) / divisorShow)));

								Vec3b corPonto;
								corPonto.val[0] = 0;
								corPonto.val[1] = 0;
								corPonto.val[2] = 255;

								Vec3b corPonto2;
								corPonto2.val[0] = 0;
								corPonto2.val[1] = 0;
								corPonto2.val[2] = 0;

								Mat MatSaveMorph;
								MatSaveMorph = fusao_mascaras_voting.clone();

								if (saiu_result == 1)
								{
									img_targetresult = img_target_big.clone();
									cout << "Aqui2" << endl;

									cvtColor(img_targetresult, img_targetresult_color, COLOR_GRAY2BGR);

									for (int k = 0; k < img_targetresult.rows; k++)
									{
										for (int l = 0; l < img_targetresult.cols; l++)
										{
											if ((double)fusao_mascaras_voting.at<unsigned char>(k, l) != 255.0)
											{
												img_targetresult.at<unsigned char>(k, l) = fusao_mascaras_voting.at<unsigned char>(k, l);
												// img_targetresult_color.at<Vec3b>(k,l) = corPonto;
												corPonto2.val[0] = img_targetresult_color.at<Vec3b>(k, l).val[0] / 3;
												corPonto2.val[1] = img_targetresult_color.at<Vec3b>(k, l).val[1] / 3;
												corPonto2.val[2] = img_targetresult_color.at<Vec3b>(k, l).val[2];

												img_targetresult_color.at<Vec3b>(k, l) = corPonto2;

												MatSaveMorph.at<unsigned char>(k, l) = (unsigned char)0;
												// imshow("MatSaveMorph", MatSaveMorph);
												printf("\n \n MatSaveMorph: linhas %d colunas %d \n \n", MatSaveMorph.rows, MatSaveMorph.cols);
											}
										}
									}
									resize(img_targetresult, img_targetresult_show, Size(((img_reference_big.cols) / divisorShow), ((img_reference_big.rows) / divisorShow)));
									// imshow("Result", img_targetresult_show);
									imwrite("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/result_color.jpg", img_targetresult_color);

									resize(img_targetresult_color, img_targetresult_color_show, Size(((img_reference_big.cols) / divisorShow), ((img_reference_big.rows) / divisorShow)));
									// imshow("Result Color", img_targetresult_color_show);
									// imwrite("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/result_color_show.jpg", img_targetresult_color_show);
									// imwrite("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/reference.jpg", img_reference_big);
									// imwrite("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/target.jpg", img_target_big);

									// waitKey(1);
									fflush(stdout);

									// Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1,1) );

									uchar *dataColor = img_targetresult_color.ptr();

									fwrite(dataColor, 1, img_targetresult_color.cols * img_targetresult_color.rows * 3, saveIMGduplaResults);

									int qtd_manchas_total_aux;
									qtd_manchas_total_aux = calcula_quantidade_manchas(fusao_mascaras_voting, threshold_multirresolucao);

									if ((save_quantidade_manchas_total = fopen("/home/luiz.tavares/Workspace/VDAO_Pixel/scripts/extra/ADMULT/results/quantidade_manchas_total.txt", "a")) == NULL)
									{ // ALTERAR DIRETORIO DA SAIDA
										printf("Arquivo de quantidade de manchas total nao pode ser aberto. \n");
										fflush(stdout);
										return (-1);
									}

									fprintf(save_quantidade_manchas_total, "%d; %d\n", contaFrameTar2, qtd_manchas_total_aux);

									fclose(save_quantidade_manchas_total);

									// para salvar a quantidade de falsos positivos e negativos por frame:
									/*
									if ((savePosNegFile = fopen("/home/gustavo/workspace/testeVideoModulo02/results/posNegFile01.txt", "a")) == NULL)
									{
										printf("Arquivo savePosNegFile01.txt nao pode ser aberto \n");
										fflush(stdout);
										return(-1);

									}

									int falPos = 0;
									int falNeg = 0;

									printf("Digite a quantidade de falsos positivos no frame: \n");
									fflush(stdout);

									fscanf(stdin, "%d", &falPos);

									printf("Digite a quantidade de falsos negativos no frame: \n");
									fflush(stdout);

									fscanf(stdin, "%d", &falNeg);

									fprintf(savePosNegFile, "%d; %d; %d\n", contaFrameTar2, falPos, falNeg);

									fclose(savePosNegFile);
									*/

									// gerando o video composto com 3 resolucoes!!!!!!!!!!!!!!!!!!!!!

									/*Mat grava3resolucoesAux0;
									Mat grava3resolucoesAux1;
									Mat grava3resolucoesAux2;

									grava3resolucoesAux0 = count_mask_vec[0].clone();
									grava3resolucoesAux1 = count_mask_vec[1].clone();
									grava3resolucoesAux2 = count_mask_vec[2].clone();

									Mat grava3resolucoesAux0color;
									Mat grava3resolucoesAux1color;
									Mat grava3resolucoesAux2color;

									cvtColor(grava3resolucoesAux0, grava3resolucoesAux0color, CV_GRAY2BGR);
									cvtColor(grava3resolucoesAux1, grava3resolucoesAux1color, CV_GRAY2BGR);
									cvtColor(grava3resolucoesAux2, grava3resolucoesAux2color, CV_GRAY2BGR);

									Mat grava3resolucoesAux0colorBig;
									Mat grava3resolucoesAux1colorBig;
									Mat grava3resolucoesAux2colorBig;*/

									// resize(grava3resolucoesAux0color, grava3resolucoesAux0colorBig, Size(((img_targetresult_color.cols)/divisorShow), ((img_targetresult_color.rows)/divisorShow)));
									/*resize(grava3resolucoesAux0color, grava3resolucoesAux0colorBig, Size(((img_targetresult_color.cols)), ((img_targetresult_color.rows))));
									resize(grava3resolucoesAux1color, grava3resolucoesAux1colorBig, Size(((img_targetresult_color.cols)), ((img_targetresult_color.rows))));
									resize(grava3resolucoesAux2color, grava3resolucoesAux2colorBig, Size(((img_targetresult_color.cols)), ((img_targetresult_color.rows))));

									Mat grava3resolucoes;
									grava3resolucoes.create(Size((img_targetresult_color.cols)*2, (img_targetresult_color.rows)*2), img_targetresult_color.type());

								  for (int k = 0; k < img_targetresult_color.rows; k++)
								  {
									  for (int l = 0; l < img_targetresult_color.cols; l++)
									  {
										grava3resolucoes.at<Vec3b>(k,l) = grava3resolucoesAux0colorBig.at<Vec3b>(k,l);

									  }
								  }

								  for (int k = 0; k < img_targetresult_color.rows; k++)
								  {
									  for (int l = img_targetresult_color.cols; l < 2*img_targetresult_color.cols; l++)
									  {
										grava3resolucoes.at<Vec3b>(k,l) = grava3resolucoesAux1colorBig.at<Vec3b>(k,l - img_targetresult_color.cols);

									  }
								  }


								  for (int k = img_targetresult_color.rows; k < 2*img_targetresult_color.rows; k++)
								  {
									  for (int l = 0; l < img_targetresult_color.cols; l++)
									  {
										grava3resolucoes.at<Vec3b>(k,l) = grava3resolucoesAux2colorBig.at<Vec3b>(k - img_targetresult_color.rows,l);

									  }
								  }

								  for (int k = img_targetresult_color.rows; k < 2*img_targetresult_color.rows; k++)
								  {
									  for (int l = img_targetresult_color.cols; l < 2*img_targetresult_color.cols; l++)
									  {
										grava3resolucoes.at<Vec3b>(k,l) = img_targetresult_color.at<Vec3b>(k - img_targetresult_color.rows,l - img_targetresult_color.cols);

									  }
							   }

								  Mat grava3resolucoesShow;

								resize(grava3resolucoes, grava3resolucoesShow, Size(((grava3resolucoes.cols)/divisorShow), ((grava3resolucoes.rows)/divisorShow)));
								*/

									// imshow("grava3resolucoesShow", grava3resolucoesShow);
									/*
																Vec3b corPontoBlack;
																corPontoBlack.val[0] = 0;
																corPontoBlack.val[1] = 0;
																corPontoBlack.val[2] = 0;

															  for (int k = 0; k < grava3resolucoesShow.rows; k++)
															  {
																  for (int l = 0; l < grava3resolucoesShow.cols; l++)
																  {
																	  if ((k == (grava3resolucoesShow.rows)/2) || (k == ((grava3resolucoesShow.rows)/2 - 1)) || (l == (grava3resolucoesShow.cols)/2) || (l == ((grava3resolucoesShow.cols)/2 - 1)))
																	  {
																		  grava3resolucoesShow.at<Vec3b>(k,l) = corPontoBlack;
																	  }

																  }
															  }
									*/

									// imshow("grava3resolucoesShow", grava3resolucoesShow);

									// uchar *dataColor3resolucoes = grava3resolucoes.ptr();
									// uchar *dataColor3resolucoes = grava3resolucoesShow.ptr();

									// fwrite(dataColor3resolucoes, 1, grava3resolucoes.cols*grava3resolucoes.rows*3, save3resolucoes);

									// fwrite(dataColor3resolucoes, 1, grava3resolucoesShow.cols*grava3resolucoesShow.rows*3, save3resolucoes);

									uchar *dataSaveMorph = MatSaveMorph.ptr();
									fwrite(dataSaveMorph, 1, MatSaveMorph.cols * MatSaveMorph.rows, saveMorph);

									// fim da geracao do video composto com 3 resolucoes!!!!!!!!!!!!!!!!!!!!!

									//(1);
									// printf("\n");
									// printf("Esperando - waitKey\n");
									// printf("\n");
									fflush(stdout);
									// waitKey(0);
								}
							}

							img_target_count_atraso = img_target_count.clone();
						}

						if (qual_idp_armazenar_aux == qual_idp_armazenar)
						{
							qual_idp_armazenar_aux = 0;
						}
						else
						{

							qual_idp_armazenar_aux = qual_idp_armazenar_aux + 1;
						}
					}
				}
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

		fclose(saveIMGduplaResults);

		// fclose(save3resolucoes);

		fclose(saveMorph);

		// fclose(save_quantidade_manchas_total);

		time_t timeEnd = time(0);
		double elapsedTime = difftime(timeEnd, timeStart);
		// LEMBRAR DE CRIAR ALGO PARA SALVAR OS ARQUIVOS
		std::cout << "elapsed time: " << elapsedTime << std::endl;

		std::ofstream myfile;
		std::stringstream streamname;
		streamname << "/home/allan.freitas/Mateus/Arquivos_Mateus/Lucas/" << currentVideo << "/elapsedTime.txt"; // ALTERAR DIRETORIO DA SAIDA
		string fileName = streamname.str();
		myfile.open(fileName.c_str());
		myfile << elapsedTime << '\n';
		myfile.close();

		//    FILE *fptime;
		//    std::stringstream streamname;
		//    streamname << "/run/media/allan/Elements/Lucas/"<<currentVideo<<"/blob/carro/high/elapsedTime.txt";
		//    string fileName = streamname.str();
		//
		//    //if ((fp = fopen("/home/gustavo/workspace/testeVideoModulo02/config/config_umbrella.txt", "r" )) == NULL)
		//    if ((fptime = fopen(fileName.c_str(), "w" )) == NULL)
		//    {
		//    	printf("Arquivo de output nao pode ser aberto. \n");
		//    	fflush(stdout);
		//    	return (-1);
		//    }
		//
		//    fprintf(fptime,"%d",elapsedTime);
		//    fclose (fptime);
	}
	return (0);
}
