# convert dat file to bin file
println("Open rating.dat and read lines")
dat   = open("./ratings.dat","r")
lines = readlines(dat)
close(dat)

l     = length(lines)
uid   = zeros(Int64,l)
mid   = zeros(Int64,l)
rat   = zeros(Int64,l)
println("Store data into numeric arrays")
for i = 1:l
    uid[i], mid[i], rat[i] = map(parse,split(lines[i],"::"))
    i%10000 == 0 &&
    println("@ line $i")
end

println("Save data into rating.bin")
fid   = open("./ratings.bin","w")
write(fid,l)
write(fid,uid,mid,rat)
close(fid)