// cube([0.1,0.1,0.1]);
// translate([0,0,0.2])cube([0.1,0.1,0.1]);
// translate([0,0.2,0])cube([0.1,0.1,0.1]);
difference(){
import("tip_left_scad.stl");
translate([-0.04,0.045,-0.2])cube([0.1,0.1,0.1]);

}

