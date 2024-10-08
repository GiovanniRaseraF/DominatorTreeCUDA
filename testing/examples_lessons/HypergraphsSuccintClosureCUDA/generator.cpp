/*
#############################################################
#															#
#				  hypergraph generator 						#
#			nvcc .\generator.cpp -o generator.exe			#
#			 cl .\generator.cpp -o generator.exe			#
#															#
#############################################################
*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <math.h>


int main(int argn, char ** argvs){
	std::srand(std::time(NULL)); // use current time as seed for random generator
	int n_v=25,m=0,n_e=n_v/2,n_s=n_v/3, to=0, m1,c, ms, remove=0;
	std::string str="", strp="";
	bool line=false, cicle=false, sphere=false;
	int conto_totale = 0, conto_temp=0, lina=0;
	
	system("cls");
	for(int i=0;i<argn-1; i++){
		str = argvs[i];
		strp = argvs[i+1];
		if(str.substr(0,2)=="-v")
			n_v=std::atoi(argvs[i+1]);
		if(str.substr(0,2)=="-e")
			n_e=std::atoi(argvs[i+1]);
		if(str.substr(0,2)=="-s")
			n_s=std::atoi(argvs[i+1]);
		if(str.substr(0,2)=="-l")
			line = true;
		if(str.substr(0,2)=="-c")
			cicle = true;
		if(str.substr(0,2)=="-o")
			sphere = true;
		if(str.substr(0,2)=="-r")
			remove = std::atoi(argvs[i+1]);
		if(str.substr(0,3)=="--h" || strp.substr(0,3)=="--h"){
			printf("Generator of hypergraph\n  generator.exe <option>\n\nOptions:\n\t-v <number of vertices>\n\t-e <number of edges>\n\t-s <number of supersets>\n\t-r <n.v.>: remove n.v. vertices\nOptions complete:\n\t-l -v <n.v.>: line graphs\n\t-l -c -v <n.v.>: line graphs cyclic\n\t-o -v <n.v.>: complete graphs");
			return 0;
		}
	}

	if(line){
		n_e = n_v -1 + (cicle? 1:0);
		n_s = n_v -1 + (cicle? 1:0);
	}else if(sphere){
		n_e = n_v * (n_v - 1);
		n_s = n_v;
	}
	
	n_e -= remove;
	
    
	printf("Inizio creazione grafo (%d %d %d)", n_v, n_e, n_s);
		
	conto_totale = n_v+n_e+n_s;
	
	std::ofstream myFile;
	std::string name_file = (!line? (sphere? "grafo_sphere_" : "grafo_") : "grafo_line_" ) + std::to_string(n_v) +"_"+ std::to_string(n_e)+"_"+std::to_string(n_s)+"o.txt";
	myFile.open(name_file.c_str());
	myFile << "INI " << n_v <<"," << n_e <<"\n";
	
	int** superset= (int**) malloc(sizeof(int*)*n_s);
	void * supSet = malloc(sizeof(std::string)*n_s);
	bool ok=true;
	int i=0;
	
	if(!line && !sphere)	
		while(i<n_s){
			ms = (std::rand()%(int)(sqrt(n_v)));
			m = 1+(std::rand()%(int)(n_v));
			m1=0,c=0;
			str="";
			for(int j=ms; j<m; j+=m1){
				str+= std::to_string(j);
				while(j+m1<=j){
					m1=(std::rand()%10);
				}
				c++;	
				if(j+m1<m) str+= ",";
				
			}
			ok=false;
			for(int j=0; j<i && !ok; j++){
				ok = ok || (((std::string*)supSet)[j]==str);
				
			}
			if(!ok && c>0){
				new (&((std::string*)supSet)[i]) std::string(str);
				superset[i] = (int*) malloc(sizeof(int)*(c+1));
				m=0; m1=fmax(str.find(","),m);
				
				for(int j=1;j<c+1; j++){
					superset[i][j] = std::stoi(str.substr(m,m1));
					m=m1+1;
					m1=str.find(",",m+1);
				}
				superset[i][0]=(c);
				
				//printf("create superset %d: %s:%d\n",i,str.c_str(),c);
				i++;
			}
			conto_temp = i;
			if((i%int(n_s/10))==0)
				printf("\033[%d;0H\t\t\t\t\t\t\t\t %d/%d",lina,conto_temp,conto_totale);
		}
	else if(line)
		for(int i=0; i<n_s; i++){
			superset[i] = new int[]{i};
			conto_temp = i;
			if(i%(n_s/10)==0)
				printf("\033[%d;0H\t\t\t\t\t\t\t\t %d/%d",lina,conto_temp,conto_totale);
		}
	else if(sphere){
		superset[0] = new int[]{0};
		printf("\033[%d;0H\t\t\t\t\t\t\t\t 1/%d",lina,conto_totale);
	}
	lina+=1;
	printf("\nwrite node");
	
	for(int i=0; i<n_v; i++){
		myFile <<  "(VE " << i <<")\n";
		conto_temp = n_s+i;
		if(i%(n_v/10)==0)
			printf("\033[%d;0H\t\t\t\t\t\t\t\t %d/%d",lina,conto_temp,conto_totale);
		
	}
	lina+=1;
	printf("\nnode write");
	
	int *remList, tempo=0, k=0;
	remList = (int*) malloc(sizeof(int)*remove);
	
	i=0;
	while(i<remove){
		ok=true;
		tempo = (std::rand()%(int)(n_e));
		for(int j=0; j<i && ok; j++){
			ok = ok && tempo!=remList[j];
		}
		if(ok){
			remList[i]=tempo;
			i++;
		}
	}
	
	std::sort(remList, remList+remove);
	
	for(int i=0; i<remove; i++) printf(" %d ",remList[i]);
	
	lina+=1;
	printf("\nedge write");
	
	if(!line && !sphere)
		for(int i=0; i<n_e; i++){
			if((k<remove && remList[k]!=i) || k>=remove){
				m = (std::rand()%(int)(n_s));
				int to = std::rand()%n_v;
				myFile << "(HA {";
				for(int j=1; j<=superset[m][0]; j+=1){
					myFile << superset[m][j];
					if(j<superset[m][0]) myFile <<  ",";
					
				}
				myFile << "}," << superset[m][0] << "," << to << ")\n";
			}else k++;
			conto_temp = n_v+n_s+i;
			if(i%(n_e/10)==0)
				printf("\033[%d;0H\t\t\t\t\t\t\t\t %d/%d",lina,conto_temp,conto_totale);
		
			
		}
	else if(line){
		
		if(cicle){
			for(int i=0; i<n_s-1; i++){
				if((k<remove && remList[k]!=i) || k>=remove){
					myFile<<"(HA {"<<i<<"},1,"<<i+1<<")\n";
				}else k++;
				conto_temp = n_v+n_s+i;
				if(i%(n_s/10)==0)
					printf("\033[%d;0H\t\t\t\t\t\t\t\t %d/%d",lina,conto_temp,conto_totale);
		
			}
			myFile<<"(HA {"<<n_s-1<<"},1,0)\n";
			conto_temp++;;
			printf("\033[%d;0H\t\t\t\t\t\t\t\t %d/%d",lina,conto_temp,conto_totale);
		
		}else{
			for(int i=0; i<n_s; i++){
				if((k<remove && remList[k]!=i) || k>=remove){
					myFile<<"(HA {"<<i<<"},1,"<<i+1<<")\n";
				}else k++;
				conto_temp = n_v+n_s+i;
				if(i%(n_s/10)==0)
					printf("\033[%d;0H\t\t\t\t\t\t\t\t %d/%d",lina,conto_temp,conto_totale);
		
			}
		}
	}else if(sphere){
		
		for(int i=0; i<n_v; i++){
			if((k<remove && remList[k]!=i) || k>=remove){
				for(int j=0; j<n_v; j++)
					if(i!=j)
						myFile<<"(HA {"<<i<<"},1,"<<j<<")\n";
			}else k++;
			conto_temp = n_v+n_s+i;
			if(i%(n_v/10)==0)
				printf("\033[%d;0H\t\t\t\t\t\t\t\t %d/%d",lina,conto_temp,conto_totale);
		}
	}
	
	myFile.close();
	
	printf("\033[%d;0H\t\t\t\t\t\t\t\t %d/%d",lina,conto_totale,conto_totale);

	
}
