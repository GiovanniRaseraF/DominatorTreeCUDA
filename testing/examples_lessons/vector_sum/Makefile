MVCCCL="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64"

vector_sum: 
	nvcc $@.cu -ccbin $(MVCCCL) -o $@_parallel.exe
	$@_parallel.exe 