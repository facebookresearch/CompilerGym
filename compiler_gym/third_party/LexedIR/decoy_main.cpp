#include "HughsLexer.h"
#include "LLLexer.h"
#include "LLToken.h"
#include "llvm_lexer.h"

#include <iostream>
#include <string>


std::string ir = "\
define dso_local i32 @_Z6squarei(i32 %0) #0 !dbg !7 {\
	%2 = alloca i32, align 4\
	store i32 %0, i32* %2, align 4\
	call void @llvm.dbg.declare(metadata i32* %2, metadata !12, metadata !DIExpression()), !dbg !13\
	%3 = load i32, i32* %2, align 4, !dbg !14\
	%4 = load i32, i32* %2, align 4, !dbg !15\
	%5 = mul nsw i32 %3, %4, !dbg !16\
	ret i32 %5, !dbg !17\
}\
\
BADGER\
\
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1\
\
attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }\
attributes #1 = { nounwind readnone speculatable willreturn }\
"

int main(){
	std::cout << "Yay!" << std::endl;
	const auto lexed = HughsLexer::LexIR(ir);
	const auto token_ids   = lexed.first;
	const auto token_atoms = lexed.second;
	return 0;
}
