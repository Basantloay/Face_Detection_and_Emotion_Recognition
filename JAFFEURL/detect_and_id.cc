#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cstring>
#include <string>

using namespace std;

int main(int argc, char *argv[]) {
	ifstream URLDataStream(argv[1]);
	string line;
	while(getline(URLDataStream, line)) {
		string URL = line.substr(3);
		string cmd = "curl \"http://apius.faceplusplus.com/v2/detection/detect?api_key=2b330a1611e4780dee14b2666c2e6ef7&api_secret=TOMk3pSLwtrBut3S7VDER6PxrxauYABl&url=" + URL + "%3Fv%3D2&attribute=smiling%2Cpose&tag=" + line.substr(0, 2) + "\"";
		system(cmd.c_str());
	}




	return 0;
}
