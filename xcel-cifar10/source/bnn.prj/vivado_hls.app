<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="bnn.prj" top="FracNet_T">
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="true" clean="true" ldflags="" mflags=""/>
    </Simulation>
    <files>
        <file name="../../bin/conv1_input.bin" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../../bin/labels.bin" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../../tb.cc" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../../weights_tb.h" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="bnn.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="bnn_tiled.cc" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="gate_mask.cc" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="gate_mask.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="conv_weights.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="dimension_def.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="layer.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="pgconv.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="typedefs.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="weights_fracnet_64.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
</AutoPilot:project>

