from nether.src.propositional import  VAR, OR, AND

def example_1():
    V0=VAR(0, "brown==1")
    V1=VAR(1, "hairless==1")
    V2=VAR(2, "solitary==1")
    C0=AND(0, [V0,V1,V2])
    V3=VAR(3, "hooves==1")
    V4=VAR(4, "small==1")
    V5=VAR(5, "furry==1")
    D0=OR(0, [V3,V4,V5])
    V6=VAR(6, "walks==1")
    C1=AND(1, [D0, V6])
    original_theory=OR(1, [C0,C1])

    V10=VAR(10, "brown==1")
    V11=VAR(11, "hairless==1")
    V12=VAR(12, "flippers==1")
    C10=AND(10, [V10,V11])
    V13=VAR(13, "hooves==1")
    V14=VAR(14, "small==1")
    V15=VAR(15, "stripes==1")
    D10=OR(10, [V13,V14])
    V16=VAR(16, "walks==1")
    C11=AND(11, [D10, V16])
    expert_theory=OR(11, [C10,C11])
    return original_theory, expert_theory


def example_2():
    V0 = VAR(0, "nocturnal==1")
    V1 = VAR(1, "newworld==1")
    V2 = VAR(2, "active==1")
    V_new=VAR(7, "strong==1")
    D_new = OR(2,[V0, V_new])
    C0 = AND(0, [D_new, V1, V2])
    V3 = VAR(3, "hooves==1")
    V4 = VAR(4, "small==1")
    V5 = VAR(5, "furry==1")
    D0 = OR(0, [V3, V4, V5])
    V6 = VAR(6, "walks==1")
    C1 = AND(1, [D0, V6])
    original_theory = OR(1, [C0, C1])

    V00 = VAR(10, "nocturnal==1")
    V10 = VAR(11, "newworld==1")
    V20 = VAR(12, "active==1")
    C00 = AND(10, [V00, V10, V20])
    V30 = VAR(13, "hooves==1")
    V40 = VAR(14, "small==1")
    V50 = VAR(15, "furry==1")
    D00 = OR(10, [V30, V40, V50])
    V60 = VAR(16, "walks==1")
    C10 = AND(11, [D00, V60])
    expert_theory = OR(1, [C00, C10])


    return original_theory, expert_theory