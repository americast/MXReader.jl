using MXReader
using Base.Test

# write your own tests here
run(`wget http://data.dmlc.ml/mxnet/models/imagenet/inception-bn.tar.gz`)
run(`tar -zxvf inception-bn.tar.gz`)
f=open("out")
line=chomp(readstring(f))

@testset "Testing Inception BN" begin
  obj = MXReader.readf("./Inception-BN-symbol.json","./Inception-BN-0126.params")
  @test length("$obj") >= 18
end
