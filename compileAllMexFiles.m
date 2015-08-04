fprintf('Make sure to set paths in code/CUDAlib/compileCUDAmex.m and code/CUDAlib/linkCUDAmex.m\n');

%% adds path to some CUDA files
addpath code/CUDAlib/

%% gets list of files to compile

fNames = compileCUDAmex();

%% compiles and links these files

for ii = 1:length(fNames);
    fprintf('Compiling file %s (%d / %d)...\n',fNames{ii},ii,length(fNames));
    compileCUDAmex(fNames{ii});
    fprintf('Linking file %s (%d / %d)...\n',fNames{ii},ii,length(fNames));
    linkCUDAmex(fNames{ii});
end

fprintf('Finished compiling and linking. Check the output for any errors.\n');
