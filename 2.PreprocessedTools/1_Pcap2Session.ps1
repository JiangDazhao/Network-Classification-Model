foreach($f in gci 1_Pcap *.pcap)
{
    0_Tool\SplitCap_2-1\SplitCap -p 100000 -b 100000 -r $f.FullName -o 2_Session\AllLayers\$($f.BaseName)-ALL
    # 0_Tool\SplitCap_2-1\SplitCap -p 100000 -b 100000 -r $f.FullName -s flow -o 2_Session\AllLayers\$($f.BaseName)-ALL
    gci 2_Session\AllLayers\$($f.BaseName)-ALL | ?{$_.Length -eq 0} | del

    #0_Tool\SplitCap_2-1\SplitCap -p 100000 -b 100000 -r $f.FullName -o 2_Session\L7\$($f.BaseName)-L7 -y L7
    # 0_Tool\SplitCap_2-1\SplitCap -p 100000 -b 100000 -r $f.FullName -s flow -o 2_Session\L7\$($f.BaseName)-L7 -y L7
    #gci 2_Session\L7\$($f.BaseName)-L7 | ?{$_.Length -eq 0} | del
}

0_Tool\finddupe -del 2_Session\AllLayers
#0_Tool\finddupe -del 2_Session\L7